# Copyright 2025 THUNKINGSPOT LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import boto3
from fastapi import UploadFile, File, HTTPException
from botocore.exceptions import BotoCoreError, ClientError
import io
import cv2
import numpy as np
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
s3 = boto3.client('s3')

logger.debug("Checking for debug mode.")
# Start the debugpy server if DEBUG_MODE is set to true
if os.getenv("DEBUG_MODE", "false").lower() == "true":
    logger.debug("Debug mode true.")
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    print("Debugger attached.")

"""
Given a list of rectangles, merge adjacent rectangles in a greedy fashion.
Each rectangle is defined as a tuple of two pixels: ((x_min, y_min), (x_max, y_max)).
Two rectangles are considered mergeable if they share a full edge (either vertical or horizontal)
so that their union is also a rectangle.

The function returns a new list of rectangles after merging. Rectangles that cannot be
merged remain in the output.
"""
def merge_adjacent_rectangles(rectangles):
    def can_merge(r1, r2):
        # r1 and r2 are in the form ((x_min, y_min), (x_max, y_max))
        (x1_min, y1_min), (x1_max, y1_max) = r1
        (x2_min, y2_min), (x2_max, y2_max) = r2

        # Check for horizontal adjacency: same vertical span and one rectangle's right edge touches the other's left edge.
        if y1_min == y2_min and y1_max == y2_max:
            if x1_max == x2_min or x2_max == x1_min:
                return True

        # Check for vertical adjacency: same horizontal span and one rectangle's bottom edge touches the other's top edge.
        if x1_min == x2_min and x1_max == x2_max:
            if y1_max == y2_min or y2_max == y1_min:
                return True

        return False

    def merge_two(r1, r2):
        # Merge two rectangles into one by taking the union.
        (x1_min, y1_min), (x1_max, y1_max) = r1
        (x2_min, y2_min), (x2_max, y2_max) = r2
        x_min = min(x1_min, x2_min)
        y_min = min(y1_min, y2_min)
        x_max = max(x1_max, x2_max)
        y_max = max(y1_max, y2_max)
        return ((x_min, y_min), (x_max, y_max))

    merged = True
    current_rectangles = rectangles[:]  # Make a copy to avoid modifying the input.
    
    # Continue merging until no further adjacent pairs are found.
    while merged:
        merged = False
        new_rectangles = []
        used = [False] * len(current_rectangles)
        for i in range(len(current_rectangles)):
            if used[i]:
                continue
            r = current_rectangles[i]
            for j in range(i + 1, len(current_rectangles)):
                if used[j]:
                    continue
                r2 = current_rectangles[j]
                if can_merge(r, r2):
                    # Merge r and r2.
                    r = merge_two(r, r2)
                    used[j] = True
                    merged = True
            new_rectangles.append(r)
            used[i] = True
        current_rectangles = new_rectangles
    
    return current_rectangles

"""
Extrudes a binary image into a 3D cuboid mesh using Trimesh. The approach is 
tolerant of noise and can handle complex shapes (spaghettification).
The image is downsampled into cuboids, and adjacent cuboids are merged into larger cuboids.

Args:
    image_path (str): Path to the binary image.
    cuboid_size (int): Minimum size of each cuboid (in X and Y dimensions). Gives
        the resolution of the final mesh.
    extrusion_height (int): Height of extrusion (in Z dimension).

Returns:
    trimesh.Trimesh: The 3D mesh object.
"""
def image_to_trimesh(binary_image, cuboid_size, extrusion_height):
    # Identify black pixels (solid regions)
    # black_pixels = np.argwhere(binary_image == 0)  # Rows and columns of black pixels
    # Downsample the image based on cuboid size
    downsampled_image = cv2.resize(
        binary_image,
        (binary_image.shape[1] // cuboid_size, binary_image.shape[0] // cuboid_size),
        interpolation=cv2.INTER_NEAREST,
    )

    # Identify solid blocks (black regions in the downsampled image)
    solid_blocks = np.argwhere(downsampled_image == 0)  # Rows and columns of black pixels

    # Turn the solid blocks into a list of rectangles
    rectangles = []
    for pixel in solid_blocks:
        x, y = pixel
        x_min = x * cuboid_size
        x_max = (x + 1) * cuboid_size
        y_min = y * cuboid_size
        y_max = (y + 1) * cuboid_size
        rectangles.append(((x_min, y_min), (x_max, y_max)))

    # Merge the rectangles where possible
    merged_rectangles = merge_adjacent_rectangles(rectangles)

    # Create a list of cuboids
    cuboids = []
    for rect in merged_rectangles:
        x_min, y_min = rect[0]
        x_max, y_max = rect[1]
        cuboids.append(trimesh.creation.box(
            extents=[x_max - x_min, y_max - y_min, extrusion_height],
            transform=trimesh.transformations.translation_matrix(
                [(x_min + x_max) / 2, (y_min + y_max) / 2, extrusion_height / 2]
            ),
        ))

    # Merge all cuboids into a single mesh
    combined_mesh = trimesh.util.concatenate(cuboids)
    # Attempt to remove duplicate internal faces (unclear if this is effective)
    combined_mesh.merge_vertices()
    combined_mesh.update_faces(combined_mesh.unique_faces())

    return combined_mesh

"""
Transforms an uploaded image file into a 3D mesh file (STL format).
Args:
    file (UploadFile): Uploaded image file.
    depth (int): Depth of extrusion (in Z dimension) in pixels.
    cuboid_size (int): Minimum pixel size of each cuboid (in X and Y dimensions).
Returns:
    io.BytesIO: The transformed 3D mesh file.
"""
def transform_image(file: UploadFile, depth: int = 100, cuboid_size: int = 1) -> io.BytesIO:
    try:
        # Validate the file type (optional)
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
        # Read the file into memory
        file_bytes = file.file.read()

        # Convert the bytes into a NumPy array
        np_array = np.frombuffer(file_bytes, np.uint8)

        # Decode the image as grayscale using OpenCV
        image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode the image. Please upload a valid image file.")

        # Convert to binary image (black and white) using a threshold
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

        # Create a 3D mesh from the binary image
        mesh = image_to_trimesh(binary_image, cuboid_size, extrusion_height=depth)
        
        # Create an STL file in memory
        stl_io = io.BytesIO()
        mesh.export(file_obj=stl_io, file_type='stl')
        stl_io.seek(0)
        return stl_io
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api")
def read_root():
    return {
        "Prompt": "This is a simple application that will render a 2-d black and "
                  "white image as a 3-d STL file."
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        xformfilename = file.filename.rsplit('.', 1)[0] + '_xform.stl'

        file.file.seek(0)
        stl_io = transform_image(file, 100, 4)

        if not stl_io:
            raise ValueError("transform_image returned None")

        # Upload the original and transformed files to S3
        file.file.seek(0)
        s3.upload_fileobj(file.file, "aquaimages-thunkingspot", file.filename)
        stl_io.seek(0)
        stl_io_copy = io.BytesIO(stl_io.getvalue())
        s3.upload_fileobj(stl_io, "aquaimages-thunkingspot", xformfilename)

        # Return the transformed file as a streaming response
        stl_io_copy.seek(0)
        response = StreamingResponse(
            stl_io_copy, 
            media_type="application/vnd.ms-pki.stl", 
            headers={"Content-Disposition": f"attachment; filename={xformfilename}"}
        )
        return response
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))