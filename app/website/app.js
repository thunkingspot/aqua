/*
  Copyright 2025 THUNKINGSPOT LLC

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

document.addEventListener('DOMContentLoaded', async () => {
  try {
    const response = await fetch('/api');
    const data = await response.json();
    document.getElementById('rootData').innerText = data.Prompt;
  } catch (error) {
    console.error('Error fetching root data:', error);
    document.getElementById('rootData').innerText = 'Error fetching root data';
  }
});

document.getElementById('uploadFile').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (file) {
    const formData = new FormData();
    formData.append('file', file, file.name);

    // Set a timeout for the API call
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 600000); // 6 min timeout

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId); // Clear the timeout if the request completes

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      // Handle the response as JSON
      const data = await response.json();
      console.log('Response data:', data); // Debugging: Log the response data to the console

      // Create an image element to display the thumbnail
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + data.thumbnail;
      img.alt = 'Thumbnail Image';
      img.style.maxWidth = '100%';
      img.style.height = 'auto';

      // Add an event listener to log errors if the image fails to load
      img.onerror = (error) => {
        console.error('Error loading image:', error); // Debugging: Log the error
      };
      
      // Append the image to a container element
      const thumbnailContainer = document.getElementById('thumbnailContainer');
      if (thumbnailContainer) {
        thumbnailContainer.innerHTML = ''; // Clear any existing content
        thumbnailContainer.appendChild(img);
        console.log('Thumbnail image appended to container'); // Debugging: Log the success
      } else {
        console.error('Thumbnail container not found'); // Debugging: Log the error
      }

      // Store the STL filename in a data attribute on the download button
      const downloadButton = document.getElementById('downloadSTL');
      downloadButton.dataset.filename = file.name;
      downloadButton.style.display = 'block'; // Show the download button

      document.getElementById('uploadStatus').innerText = 'Thumbnail image displayed successfully';
    } catch (error) {
      console.error('Error uploading file:', error);
      document.getElementById('uploadStatus').innerText = 'Error uploading file';
    }
  }
});

// Add an event listener for the download button
document.getElementById('downloadSTL').addEventListener('click', async () => {
  const downloadButton = document.getElementById('downloadSTL');
  const imageFileName = downloadButton.dataset.filename;

  if (imageFileName) {
      try {
          const stlResponse = await fetch(`/api/download/${imageFileName}`);
          if (!stlResponse.ok) {
              throw new Error('Network response was not ok');
          }
          const stlBlob = await stlResponse.blob();

          // Create a download link for the STL file
          const stlUrl = URL.createObjectURL(stlBlob);
          const a = document.createElement('a');
          a.href = stlUrl;
          const contentDisposition = stlResponse.headers.get('Content-Disposition');
          const filename = contentDisposition ? contentDisposition.split('filename=')[1] : 'download.stl';
          a.download = filename.replace(/"/g, ''); // Remove any surrounding quotes
          a.click();
          console.log('STL file saved to disk'); // Debugging: Log the success

          // Clean up the object URL
          URL.revokeObjectURL(stlUrl);
      } catch (error) {
          console.error('Error downloading STL file:', error);
      }
  } else {
      console.error('STL filename not found');
  }
});
