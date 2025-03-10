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

      // Handle the response as a Blob
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;

      // Extract the filename from the Content-Disposition header
      const contentDisposition = response.headers.get('Content-Disposition');
      let fileName = 'downloaded_file.stl';
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="?(.+)"?/);
        if (match && match[1]) {
          fileName = match[1];
        }
      }

      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.getElementById('uploadStatus').innerText = 'File transformed and downloaded successfully';
    } catch (error) {
      console.error('Error uploading file:', error);
      document.getElementById('uploadStatus').innerText = 'Error uploading file';
    }
  }
});
