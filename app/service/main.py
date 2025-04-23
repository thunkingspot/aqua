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
import os
import sys
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import boto3
from fastapi import UploadFile, File, HTTPException
from botocore.exceptions import BotoCoreError, ClientError
import io
import cv2
import numpy as np
import scipy
import trimesh
import moderngl
from PIL import Image
from pyrr import Matrix44
import base64
import logging
from logging.handlers import RotatingFileHandler
import debugpy

# Log directory created during deployment exists
log_directory = './log'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure logging to output to console and file
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

file_handler = RotatingFileHandler(
    os.path.join('app.log'), maxBytes=5*1024*1024, backupCount=3  # 5 MB per file, 3 backup files
)
file_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])
logger = logging.getLogger(__name__)

# Set boto3 and botocore logging level to WARNING to prevent logging sensitive information
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)

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
    
    # Merge vertices that are at the same position with higher precision
    combined_mesh.merge_vertices(digits_vertex=8)  # Higher precision for better merging
    
    # Process the mesh to ensure proper topology
    combined_mesh.process(validate=True)
    
    # Fix normal directions and recalculate them
    combined_mesh.fix_normals(multibody=True)
    combined_mesh.vertex_normals = trimesh.geometry.mean_vertex_normals(
        len(combined_mesh.vertices),
        combined_mesh.faces,
        combined_mesh.face_normals,
        weight='area'
    )
    
    # Update the mesh with all vertices
    vertex_mask = np.ones(len(combined_mesh.vertices), dtype=bool)
    combined_mesh.update_vertices(vertex_mask)
    combined_mesh.update_faces(np.ones(len(combined_mesh.faces), dtype=bool))

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
        logger.debug(f"Reading file: {str(file.filename)}")
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
        logger.debug(f"Extruding file: {str(file.filename)}")
        mesh = image_to_trimesh(binary_image, cuboid_size, extrusion_height=depth)
        return mesh
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

"""
Generate an stil file in memory from a 3d mesh object.
Args:
    mesh (trimesh.Trimesh): The 3d mesh object.
Returns:
    io.BytesIO: The STL file.
"""
def generate_stl(mesh):
    # Create an STL file in memory
    stl_io = io.BytesIO()
    mesh.export(file_obj=stl_io, file_type='stl')
    stl_io.seek(0)
    return stl_io

"""
Generate a thumbnail of the 3d mesh object.
The thumbnail is a 2d image of the mesh object shown from a 3/4 perspective.
Args:
    mesh (trimesh.Trimesh): The 3d mesh object.
    width (int): Width of the thumbnail image.
    height (int): Height of the thumbnail image.
Returns:
    io.BytesIO: The thumbnail image file.
"""
def generate_thumbnail(mesh, width=640, height=480):
    mesh.rezero()
    
    # Calculate scaling factor to normalize mesh size
    size = mesh.bounds[1] - mesh.bounds[0]
    max_dimension = np.max(size)
    scale_factor = 10.0 / max_dimension  # Normalize to 10 units
    
    # Create a scaled copy of the mesh
    scaled_mesh = mesh.copy()
    scaled_mesh.apply_scale(scale_factor)

    # Load trimesh mesh and extract geometry
    vertices = np.array(scaled_mesh.vertices, dtype=np.float32)
    faces = np.array(scaled_mesh.faces, dtype=np.uint32).flatten()
    normals = np.array(scaled_mesh.vertex_normals, dtype=np.float32)

    # Create an offscreen context.
    # ModernGL can create a context for offscreen rendering using its standalone Context.
    ctx = moderngl.create_standalone_context()

    # Create a framebuffer.
    fbo = ctx.simple_framebuffer((width, height))
    fbo.use()

    # Define vertex and fragment shaders for ModernGL.
    # For OpenGL ES 3.2 or OpenGL 3.3 compatibility, you might use modern GLSL.
    vertex_shader_src = '''
    #version 330
    in vec3 in_position;
    in vec3 in_normal;
    uniform mat4 mvp;
    uniform mat4 model;
    out vec3 frag_normal;
    out vec3 frag_position;
    void main() {
        frag_normal = mat3(transpose(inverse(model))) * in_normal;
        frag_position = vec3(model * vec4(in_position, 1.0));
        gl_Position = mvp * vec4(in_position, 1.0);
    }
    '''

    fragment_shader_src = '''
    #version 330
    in vec3 frag_normal;
    in vec3 frag_position;
    out vec4 fragColor;
    uniform vec3 light_position;
    uniform vec3 view_position;
    void main() {
        // Normalize the normal vector
        vec3 normal = normalize(frag_normal);

        // Calculate the light direction and distance
        vec3 light_dir = normalize(light_position - frag_position);
        float light_distance = length(light_position - frag_position);
        
        // Softer falloff for area light simulation
        float falloff = 1.0 / (1.0 + 0.01 * light_distance * light_distance);  // Reduced falloff coefficient
        
        // Calculate the view direction
        vec3 view_dir = normalize(view_position - frag_position);

        // Calculate the reflection direction with softening
        vec3 reflect_dir = reflect(-light_dir, normal);
        float spread = 0.01;
        reflect_dir = normalize(reflect_dir + spread * (view_dir - reflect_dir));

        // Calculate the ambient, diffuse, and specular components
        float ambient_strength = 0.25;  // Increased ambient light
        vec3 ambient = ambient_strength * vec3(1.0, 1.0, 1.0);

        // Very soft diffuse falloff
        float diff = pow(max(dot(normal, light_dir), 0.0), 0.9);
        vec3 diffuse = diff * vec3(8.0, 8.0, 8.0) * falloff;  // Increased diffuse intensity

        // Much wider specular highlights
        float specular_strength = 2.1;  // Increased specular intensity
        float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 10.0);
        vec3 specular = specular_strength * spec * vec3(1.8, 1.8, 1.8) * falloff;

        // Combine the components with smooth transition and increased overall brightness
        vec3 result = (ambient + diffuse + specular) * vec3(0.2, 0.8, 0.6);  // Increased base color intensity
        fragColor = vec4(result, 1.0);
    }
    '''

    # Compile the shader program.
    prog = ctx.program(
        vertex_shader=vertex_shader_src,
        fragment_shader=fragment_shader_src,
    )

    # Create a Vertex Buffer Object (VBO) for the vertex data.
    vbo = ctx.buffer(vertices.tobytes())

    # Create a Vertex Buffer Object (VBO) for the normal data.
    nbo = ctx.buffer(normals.tobytes())

    # Create an Element Buffer Object (EBO) for the indices.
    ebo = ctx.buffer(faces.tobytes())

    # Define the format for the vertex attributes.
    vao_content = [
        # Format: (buffer, 'format string', 'attribute name')
        (vbo, '3f', 'in_position'),
        (nbo, '3f', 'in_normal'),
    ]

    # Create a Vertex Array Object (VAO) binding the VBO and the shader program.
    vao = ctx.vertex_array(prog, vao_content, ebo)

    # Calculate the bounding box of the mesh
    bounding_box = scaled_mesh.bounds
    center = bounding_box.mean(axis=0)
    size = bounding_box[1] - bounding_box[0]

    # Set the camera position for a consistent 3/4 view
    diagonal = np.linalg.norm(size)  # Get the diagonal length
    camera_distance = diagonal * 1.0  # Use diagonal for consistent distance
    eye = center + np.array([1.0, 1.0, 0.8]) * camera_distance  # Fixed ratio for 3/4 view

    # Adjust near and far clipping planes based on the diagonal
    near_plane = camera_distance * 0.01
    far_plane = camera_distance * 4.0

    # Set up the Model-View-Projection (MVP) matrix
    view = Matrix44.look_at(
        eye=eye,
        target=center,
        up=np.array([0, 0, 1])
    )

    projection = Matrix44.perspective_projection(45, width / height, near_plane, far_plane)
    model = Matrix44.identity()
    mvp = projection * view * model
    prog['mvp'].write(mvp.astype('f4').tobytes())
    prog['model'].write(model.astype('f4').tobytes())

    # Calculate light position based on object size
    light_scale = np.linalg.norm(size) * 0.75  # Scale factor for light position
    light_position = eye + np.array([light_scale, light_scale * 0.7, light_scale * 0.7])
    prog['light_position'].write(light_position.astype('f4').tobytes())
    prog['view_position'].write(eye.astype('f4').tobytes())

    # Clear the framebuffer and render the scene.
    ctx.clear(1.0, 1.0, 1.0, 1.0)  # clear to white
    vao.render()

    # Read the framebuffer into a numpy array.
    pixel_data = fbo.read(components=4, alignment=1)

    # Read pixels from the FBO
    image = Image.frombytes("RGBA", (width, height), pixel_data)

    # OpenGL's origin is bottom-left; flip the image vertically.
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Save the image to a file (for debugging)
    image.save("/home/ubuntu/Downloads/thumbnail.png")

    # Convert the image to a bytes stream
    image_io = io.BytesIO()
    image.save(image_io, format='PNG')
    image_io.seek(0)
    return image_io

"""
Root endpoint that provides a simple description of the API.
"""
@app.get("/api")
def read_root():
    return {
        "Prompt": "This is a simple application that will render a 2-d black and "
                  "white image as a 3-d mesh by extruding the image to a given depth."
    }

"""
Upload endpoint that accepts an image file creates a 3-d mesh from an "extrusion" of the image.
Args:
    file (UploadFile): The uploaded image file.
Returns:
    StreamingResponse: The thumbnail image file.
Side Effects:
    Uploads the original and transformed files to an S3 bucket
    with names of the form "<filename>.png" and "<filename>_xform.stl".
"""
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        xformfilename = file.filename.rsplit('.', 1)[0] + '_xform.stl'

        file.file.seek(0)
        logger.debug(f"Generating 3d mesh: {str(file.filename)}")
        mesh = transform_image(file, 100, 4)
        logger.debug(f"Converting to STL: {str(file.filename)}")
        stl_io = generate_stl(mesh)
        logger.debug(f"Generating thumbnail: {str(file.filename)}")
        thumbnail = generate_thumbnail(mesh)

        if not stl_io:
            raise ValueError("transform_image returned None")

        # Upload the original and transformed files to S3
        file.file.seek(0)
        s3.upload_fileobj(file.file, "aquaimages-thunkingspot", file.filename)
        stl_io.seek(0)
        stl_io_copy = io.BytesIO(stl_io.getvalue())
        s3.upload_fileobj(stl_io, "aquaimages-thunkingspot", xformfilename)

        # Convert the thumbnail image to a base64-encoded string
        thumbnail.seek(0)
        base64_thumbnail = base64.b64encode(thumbnail.read()).decode('utf-8')

        # Return the base64-encoded image as a JSON response
        return {"thumbnail": base64_thumbnail}
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Boto3 error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        logger.error(f"HTTP error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
"""
Download the STL file from s3 associated with the given filename of the originally
transformed 2-d image.
Args:
    filename (str): The filename of the original 2-d image.
Returns:
    StreamingResponse: The STL file.
"""
@app.get("/api/download/{filename}")
def download_stl(filename: str):
    try:
        xformfilename = filename.rsplit('.', 1)[0] + '_xform.stl'
        logger.debug(f"Downloading file: {xformfilename}")
        stl_io = io.BytesIO()
        s3.download_fileobj("aquaimages-thunkingspot", xformfilename, stl_io)
        stl_io.seek(0)
        return StreamingResponse(
            stl_io, 
            media_type="application/vnd.ms-pki.stl", 
            headers={"Content-Disposition": f"attachment; filename={xformfilename}"}
        )
   
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

