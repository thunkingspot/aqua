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

import io
import struct
from unittest.mock import Mock, patch
from contextlib import contextmanager
from fastapi import File, HTTPException
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from app.service.main import app, transform_image, generate_stl, generate_thumbnail
from PIL import Image
import trimesh

# Create a mock image for testing - a black square in the center of a white background
def create_mock_simple_png_image(edge_length: int=15, square_size: int=5):
    # Create a image with size edge_length with a black square of square_size
    # in the center with a white border
    image = Image.new("RGB", (edge_length, edge_length), "white")
    start = (edge_length - square_size) // 2
    end = start + square_size
    for x in range(start, end):
        for y in range(start, end):
            image.putpixel((x, y), (0, 0, 0))

    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()

# Create a mock image for testing
# edge_length: size of the image
# diameter: diameter of the circle in the center of the image
# stroke_width: width of the circle's stroke
def create_mock_complex_png_image(edge_length: int=15, diameter: int=5, stroke_width: int=1):
    # Create a image with size edge_length with a circle in the center
    # with a white border
    image = Image.new("RGB", (edge_length, edge_length), "white")

    # Calculate the circle's bounding box
    start = (edge_length - diameter) // 2
    end = start + diameter

    # Draw the circle with a white border
    for x in range(start, end):
        for y in range(start, end):
            # Calculate the distance from the center of the circle
            distance = ((x - edge_length // 2) ** 2 + (y - edge_length // 2) ** 2) ** 0.5
            if diameter // 2 - stroke_width <= distance <= diameter // 2:
                image.putpixel((x, y), (0, 0, 0))

    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()

# Create mock of uploaded file
def create_mock_uploaded_file(filename: str = "test.png", type: str = "image/png", content: bytes = b""):
    # Create a mock file object
    mock_file = Mock()
    mock_file.filename = filename
    mock_file.content_type = type
    if not content:
        content = create_mock_simple_png_image()
    mock_file.file = io.BytesIO(content)
    return mock_file


# Test the transform_image function happy path
def test_transform_image_success():
    # should produce single voxel with 6 faces and 2 triangles per face
    image_size = 15
    square_size = 5
    extruded_depth = 5
    voxel_size = 5

    # Mock the file read method to return the image content
    mock_file = create_mock_uploaded_file(
        filename="test.png",
        type="image/png",
        content=create_mock_simple_png_image(image_size, square_size))

    # Call the transform_image function with the mocked file
    mesh = transform_image(mock_file, extruded_depth, voxel_size)

    # Assert that the result is not None and is an instance of io.BytesIO
    assert mesh is not None
    assert isinstance(mesh, trimesh.Trimesh)

    # A square size of 5 with a voxel size of 5 should result in one voxel
    # with 6 faces and 2 triangles per face
    triangle_count = len(mesh.faces)
    assert triangle_count == 12

# Test the transform_image function with multiple cuboids
def test_transform_image_success_multiple_cuboids():
    # should still produce 6 faces and 2 triangles per face even though
    # there are multiple cuboids
    image_size = 15
    square_size = 10
    extruded_depth = 5
    voxel_size = 5

    # Mock the file read method to return the image content
    mock_file = create_mock_uploaded_file(
        filename="test.png",
        type="image/png",
        content=create_mock_simple_png_image(image_size, square_size))

    # Call the transform_image function with the mocked file
    mesh = transform_image(mock_file, extruded_depth, voxel_size)

    # 6 faces and 2 triangles per face
    triangle_count = len(mesh.faces)
    assert triangle_count == 12

def test_transform_image_success_spaghetti():
    # should still produce 6 faces and 2 triangles per face even though
    # there are multiple voxels
    image_size = 50
    diameter = 40
    stroke_width = 10
    extruded_depth = 5
    voxel_size = 5

    # Mock the file read method to return the image content
    mock_file = create_mock_uploaded_file(
        filename="test.png",
        type="image/png",
        content=create_mock_complex_png_image(image_size, diameter, stroke_width))

    # Call the transform_image function with the mocked file
    mesh = transform_image(mock_file, extruded_depth, voxel_size)

    # Assert that the result is not None and is an instance of io.BytesIO
    assert mesh is not None

    # ad hoc derived assertion - it will tell us if changes to the algorithm
    # change the behavior of the function
    triangle_count = len(mesh.faces)
    assert triangle_count == 132

# Test the transform_image function with an invalid image
def test_transform_image_invalid_image():
    # Create a mock file object with an invalid content type
    mock_file = create_mock_uploaded_file(filename="test.txt", type="text/plain")

    # Call the transform_image function with the mocked file
    try:
        transform_image(mock_file)
        assert False, "transform_image did not raise an exception for an invalid image"
    except HTTPException as e:
        assert e.status_code == 400

# Test the transform_image function with an image that cannot be decoded
def test_transform_image_invalid_image_decode():
    # Create a mock file object with an invalid image content
    mock_file = create_mock_uploaded_file(content=b"invalid content")

    # Call the transform_image function with the mocked file
    try:
        transform_image(mock_file)
        assert False, "transform_image did not raise an exception for an image that cannot be decoded"
    except HTTPException as e:
        assert e.status_code == 400

client = TestClient(app)

# Test generating a thumbnail image of the 3d mesh
def test_generate_thumbnail():
    # should produce single voxel with 6 faces and 2 triangles per face
    image_size = 15
    square_size = 5
    extruded_depth = 5
    voxel_size = 5

    # Mock the file read method to return the image content
    mock_file = create_mock_uploaded_file(
        filename="test.png",
        type="image/png",
        content=create_mock_simple_png_image(image_size, square_size))

    # Call the transform_image function with the mocked file
    mesh = transform_image(mock_file, extruded_depth, voxel_size)

    # Assert that the result is not None and is an instance of io.BytesIO
    assert mesh is not None

    # ad hoc derived assertion - it will tell us if changes to the algorithm
    # change the behavior of the function
    triangle_count = len(mesh.faces)
    assert triangle_count == 12

    thumbnail = generate_thumbnail(mesh, 200, 200)

    # Assert that the thumbnail is not None and is an instance of io.BytesIO
    assert thumbnail is not None
    assert isinstance(thumbnail, io.BytesIO)

    # Assert that the thumbnail is a PNG image
    thumbnail_image = Image.open(thumbnail)
    assert thumbnail_image.format == "PNG"

# Test generating an stl file from the 3d mesh
def test_generate_stl():
    # should produce single voxel with 6 faces and 2 triangles per face
    image_size = 15
    square_size = 5
    extruded_depth = 5
    voxel_size = 5

    # Mock the file read method to return the image content
    mock_file = create_mock_uploaded_file(
        filename="test.png",
        type="image/png",
        content=create_mock_simple_png_image(image_size, square_size))

    # Call the transform_image function with the mocked file
    mesh = transform_image(mock_file, extruded_depth, voxel_size)

    # Generate an stl file from the 3d mesh
    stl = generate_stl(mesh)

    # Assert that the stl is not None and is an instance of io.BytesIO
    assert stl is not None
    assert isinstance(stl, io.BytesIO)

    stl_content = stl.getvalue()

    # Assert that the content of the stl is as expected
    # Check for the presence of the binary STL header (80 bytes) and triangle count (4 bytes)
    assert len(stl_content) > 84  # Header (80 bytes) + Triangle count (4 bytes)
    header = stl_content[:80]
    triangle_count = struct.unpack('<I', stl_content[80:84])[0]

    # Print the header and triangle count for debugging
    # print(f"Header: {header}")
    # print(f"Triangle count: {triangle_count}")

    # A square size of 5 with a voxel size of 5 should result in one voxel
    # with 6 faces and 2 triangles per face
    assert triangle_count == 12

# Test @app.get("/api") from main.py
def test_read_root():
    response = client.get("/api")
    assert response.status_code == 200
    assert "3-d mesh" in response.json().get("Prompt", "")

# Test @app.post("/api/upload") from main.py - happy path
# mock S3 and transform methods and stl_io.getvalue()
@patch('app.service.main.s3.upload_fileobj')
@patch('app.service.main.transform_image')
@patch('app.service.main.generate_stl')
@patch('app.service.main.generate_thumbnail')
def test_upload_file(mock_generate_thumbnail, mock_generate_stl, mock_transform_image, mock_upload_fileobj):
    mock_generate_thumbnail.return_value = io.BytesIO(b"mocked thumbnail content")
    mock_transform_image.return_value = Mock()
    mock_transform_image.return_value.getvalue.return_value = b"mocked mesh content"
    mock_generate_stl.return_value = io.BytesIO(b"mocked stl content")
    response = client.post("/api/upload", files={"file": ("test.png", b"test")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"
    assert response.content == b"mocked thumbnail content"
    
# Test @app.post("/api/upload") from main.py - transform_image returns None
@patch('app.service.main.transform_image')
def test_upload_file_transform_image_none(mock_transform_image):
    mock_transform_image.return_value = None
    response = client.post("/api/upload", files={"file": ("test.png", b"test")})
    assert response.status_code == 500
    assert "None" in response.json()["detail"]

# Test @app.post("/api/upload") from main.py - invalid image
@patch('app.service.main.transform_image')
def test_upload_file_invalid_image(mock_transform_image):
    mock_transform_image.side_effect = HTTPException(status_code=400, detail="mock msg")
    response = client.post("/api/upload", files={"file": ("test.txt", b"invalid content")})
    assert response.status_code == 400
    assert response.json() == {"detail": "mock msg"}

# Test @app.post("/api/upload") from main.py - s3 upload error
@patch('app.service.main.s3.upload_fileobj')
@patch('app.service.main.transform_image')
def test_upload_file_s3_error(mock_transform_image, mock_upload_fileobj):
    mock_transform_image.return_value = Mock()
    mock_upload_fileobj.side_effect = ClientError({"Error": {"Code": "500", "Message": "mock msg"}}, "upload_fileobj")
    response = client.post("/api/upload", files={"file": ("test.png", b"test")})
    assert response.status_code == 400
    assert "500" in response.json()["detail"]
    assert "mock msg" in response.json()["detail"]

# Test @app.post("/api/upload") from main.py - unexpected error
@patch('app.service.main.s3.upload_fileobj')
@patch('app.service.main.transform_image')
def test_upload_file_unexpected_error(mock_transform_image, mock_upload_fileobj):
    mock_transform_image.return_value = Mock()
    mock_upload_fileobj.side_effect = Exception("Unexpected error")
    response = client.post("/api/upload", files={"file": ("test.png", b"test")})
    assert response.status_code == 500
    assert response.json() == {"detail": "Unexpected error"}

# Test @app.post("/api/download") from main.py - happy path
@patch('app.service.main.s3.download_fileobj')
def test_download_file(mock_download_fileobj):
    mock_download_fileobj.side_effect = lambda Bucket, Key, Fileobj: \
        Fileobj.write(b"mocked stl content")
    response = client.get("/api/download/test.png")
    assert response.status_code == 200
    assert response.headers["Content-Disposition"] == "attachment; filename=test_xform.stl"
    assert response.headers["Content-Type"] == "application/vnd.ms-pki.stl"
    assert response.content == b"mocked stl content"

# Test @app.get("/api/download/{filename}") from main.py - file not found
@patch('app.service.main.s3.download_fileobj')
def test_download_file_not_found(mock_download_fileobj):
    mock_download_fileobj.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "download_fileobj")
    response = client.get("/api/download/test.png")
    assert response.status_code == 400
    assert "404" in response.json()["detail"]
    assert "Not Found" in response.json()["detail"]

# Test @app.get("/api/download/{filename}") from main.py - s3 download error
@patch('app.service.main.s3.download_fileobj')
def test_download_file_s3_error(mock_download_fileobj):
    mock_download_fileobj.side_effect = ClientError({"Error": {"Code": "500", "Message": "mock msg"}}, "download_fileobj")
    response = client.get("/api/download/test.png")
    assert response.status_code == 400
    assert "500" in response.json()["detail"]
    assert "mock msg" in response.json()["detail"]

# Test @app.get("/api/download/{filename}") from main.py - unexpected error
@patch('app.service.main.s3.download_fileobj')
def test_download_file_unexpected_error(mock_download_fileobj):
    mock_download_fileobj.side_effect = Exception("Unexpected error")
    response = client.get("/api/download/test.png")
    assert response.status_code == 500
    assert response.json() == {"detail": "Unexpected error"}
