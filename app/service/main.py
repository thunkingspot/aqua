from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import boto3
from fastapi import UploadFile, File, HTTPException
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image
import io
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from shapely.ops import triangulate
import trimesh
import logging
import debugpy
import os

# Configure logging to output to console
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a FastAPI app
app = FastAPI()

# Create a Boto3 client for interacting with S3 and dynamodb
#dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
#s3 = boto3.client('s3')

logger.debug("Checking for debug mode.")
# Start the debugpy server if DEBUG_MODE is set to true
if os.getenv("DEBUG_MODE", "false").lower() == "true":
    logger.debug("Debug mode true.")
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    print("Debugger attached.")

def bitmap_to_triangles_with_holes(binary_image, epsilon=1.0):
    # Process a binary image to extract contours and triangulate the black regions.
    # Returns both the triangles and the contours.

    # Find contours (external and internal)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return [], []  # No contours found

    all_triangles = []
    all_contours = []

    # Identify the outer boundary (white region)
    for idx, contour in enumerate(contours):
        is_external = hierarchy[0][idx][3] == -1  # Contour with no parent
        if not is_external:
            continue  # Skip inner contours

        # Simplify the outer boundary
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        outer_ring = LinearRing(simplified_contour[:, 0, :])

        # Identify black regions within the outer boundary
        holes = []
        child_idx = hierarchy[0][idx][2]  # First child of the outer boundary
        while child_idx != -1:
            child_contour = contours[child_idx]
            simplified_child = cv2.approxPolyDP(child_contour, epsilon, True)
            holes.append(LinearRing(simplified_child[:, 0, :]))
            child_idx = hierarchy[0][child_idx][0]  # Next sibling contour

        # Create the outer polygon with holes
        polygon = Polygon(outer_ring, holes)
        if polygon.is_valid:
            all_contours.append(polygon)

            # Triangulate the polygon
            polygon_triangles = triangulate(polygon)
            for triangle in polygon_triangles:
                all_triangles.append(np.array(triangle.exterior.coords[:-1]))

    return all_contours, all_triangles

def extrude_2d_to_3d(triangles_2d, contours, extrusion_depth):
    vertices = []
    faces = []

    # Create top and bottom surfaces
    for triangle in triangles_2d:
        # Bottom surface (Z=0)
        base_index = len(vertices)
        vertices.extend([[x, y, 0] for x, y in triangle])

        # Top surface (Z=extrusion_depth)
        vertices.extend([[x, y, extrusion_depth] for x, y in triangle])

        # Top and bottom triangles
        faces.extend([
            [base_index, base_index + 1, base_index + 2],  # Bottom triangle
            [base_index + 3, base_index + 4, base_index + 5],  # Top triangle
        ])

    # Create vertical walls for contours
    for contour in contours:
        points = np.array(contour.exterior.coords[:-1])  # Contour points

        for i in range(len(points)):
            # Connect current point to the next (loop back at the end)
            next_index = (i + 1) % len(points)

            # Bottom and top vertices
            bottom_start = len(vertices)
            vertices.append([points[i][0], points[i][1], 0])  # Bottom current
            vertices.append([points[next_index][0], points[next_index][1], 0])  # Bottom next
            vertices.append([points[i][0], points[i][1], extrusion_depth])  # Top current
            vertices.append([points[next_index][0], points[next_index][1], extrusion_depth])  # Top next

            # Create two triangles for the quad
            faces.extend([
                [bottom_start, bottom_start + 1, bottom_start + 2],  # First triangle
                [bottom_start + 1, bottom_start + 3, bottom_start + 2],  # Second triangle
            ])

    return np.array(vertices), np.array(faces)


def spaghetti_image_to_trimesh(binary_image, voxel_size=1.0, extrusion_height=10.0):
    """
    Extrudes a spaghetti-style binary image into a 3D voxel mesh with adjacent voxels merged using Trimesh.

    Args:
        image_path (str): Path to the binary image.
        voxel_size (float): Size of each voxel (in X and Y dimensions).
        extrusion_height (float): Height of extrusion (in Z dimension).

    Returns:
        trimesh.Trimesh: The 3D mesh object.
    """

    # Identify black pixels (solid regions)
    #black_pixels = np.argwhere(binary_image == 0)  # Rows and columns of black pixels
    # Downsample the image based on voxel size
    block_size = int(voxel_size)  # Size of each block in pixels
    downsampled_image = cv2.resize(
        binary_image,
        (binary_image.shape[1] // block_size, binary_image.shape[0] // block_size),
        interpolation=cv2.INTER_NEAREST,
    )

    # Identify solid blocks (black regions in the downsampled image)
    solid_blocks = np.argwhere(downsampled_image == 0)  # Rows and columns of black pixels

    # Create a list of cuboids (voxels)
    voxels = []
    for pixel in solid_blocks:
        x, y = pixel  # Pixel coordinates in the 2D image

        # Define voxel bounds
        x_min = x * voxel_size
        x_max = (x + 1) * voxel_size
        y_min = y * voxel_size
        y_max = (y + 1) * voxel_size
        z_min = 0
        z_max = extrusion_height

        # Create a cuboid for the voxel
        voxel = trimesh.creation.box(
            extents=[voxel_size, voxel_size, extrusion_height],
            transform=trimesh.transformations.translation_matrix(
                [(x_min + x_max) / 2, (y_min + y_max) / 2, extrusion_height / 2]
            ),
        )
        voxels.append(voxel)

    # Merge all voxels into a single mesh
    combined_mesh = trimesh.util.concatenate(voxels)

    # Remove duplicate internal faces
    combined_mesh.merge_vertices()
    combined_mesh.remove_duplicate_faces()

    return combined_mesh

def transform_image(file: UploadFile, depth: int = 100, tolerance: float = 2.5) -> io.BytesIO:
    try:
        # Validate the file type (optional)
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
        # Read the file into memory
        file_bytes = file.file.read()

        # Convert the bytes into a NumPy array
        np_array = np.frombuffer(file_bytes, np.uint8)

        # Decode the image using OpenCV
        image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode the image. Please upload a valid image file.")

        # Process the image (example: convert to binary)
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

        mesh = spaghetti_image_to_trimesh(binary_image, voxel_size=4.0, extrusion_height=depth)

        # Create an STL file in memory
        stl_io = io.BytesIO()
        mesh.export(file_obj=stl_io, file_type='stl')
        stl_io.seek(0)
        return stl_io
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api")
def read_root():
    return {"Hello": "World /"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        xformfilename = file.filename.rsplit('.', 1)[0] + '_xform.stl'

        file.file.seek(0)
        stl_io = transform_image(file, 100, 0.5)

        if not stl_io:
            raise ValueError("transform_image returned None")

        file.file.seek(0)
        s3.upload_fileobj(file.file, "aquaimages-thunkingspot", file.filename)

        logger.debug(f"STL buffer size before seek: {stl_io.getbuffer().nbytes}")
        stl_io.seek(0)
        # Create a copy of the BytesIO object
        stl_io_copy = io.BytesIO(stl_io.getvalue())
        logger.debug(f"STL buffer size after seek: {stl_io.getbuffer().nbytes}")
        s3.upload_fileobj(stl_io, "aquaimages-thunkingspot", xformfilename)

        # Return the transformed file as a streaming response
        logger.debug(f"STL buffer size before seek: {stl_io_copy.getbuffer().nbytes}")
        stl_io_copy.seek(0)
        logger.debug(f"STL buffer size after seek: {stl_io_copy.getbuffer().nbytes}")
        response = StreamingResponse(
            stl_io_copy, 
            media_type="application/vnd.ms-pki.stl", 
            headers={"Content-Disposition": f"attachment; filename={xformfilename}"}
        )
        return response
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))