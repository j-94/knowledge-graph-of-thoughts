import fs from 'fs';
import { uploadToR2, getFromR2, listR2Objects, deleteFromR2 } from './r2-operations.js';

/**
 * Example function to upload a file to R2
 * @param {string} filePath - Path to the local file
 * @param {string} destination - Destination key in R2
 */
async function uploadFileExample(filePath, destination) {
  try {
    // Read the file
    const fileContent = fs.readFileSync(filePath);
    
    // Detect content type (or set manually)
    const contentType = detectContentType(filePath);
    
    // Upload to R2
    const result = await uploadToR2(destination, fileContent, contentType);
    
    if (result.success) {
      console.log(`Successfully uploaded ${filePath} to ${destination}`);
      console.log(`ETag: ${result.etag}`);
    } else {
      console.error(`Failed to upload: ${result.error}`);
    }
  } catch (error) {
    console.error('Error in upload example:', error);
  }
}

/**
 * Example function to list all objects in a directory
 * @param {string} prefix - Optional prefix/directory to list
 */
async function listObjectsExample(prefix = '') {
  try {
    const result = await listR2Objects(prefix);
    
    if (result.success) {
      console.log(`Found ${result.objects.length} objects:`);
      result.objects.forEach(obj => {
        console.log(`- ${obj.Key} (${formatBytes(obj.Size)})`);
      });
    } else {
      console.error(`Failed to list objects: ${result.error}`);
    }
  } catch (error) {
    console.error('Error in list example:', error);
  }
}

/**
 * Example function to download a file from R2
 * @param {string} key - Object key in R2
 * @param {string} destination - Local destination path
 */
async function downloadFileExample(key, destination) {
  try {
    const result = await getFromR2(key);
    
    if (result.success) {
      // Handle the stream
      const chunks = [];
      for await (const chunk of result.data) {
        chunks.push(chunk);
      }
      
      // Write to local file
      fs.writeFileSync(destination, Buffer.concat(chunks));
      console.log(`Successfully downloaded ${key} to ${destination}`);
    } else {
      console.error(`Failed to download: ${result.error}`);
    }
  } catch (error) {
    console.error('Error in download example:', error);
  }
}

/**
 * Example function to delete an object from R2
 * @param {string} key - Object key to delete
 */
async function deleteObjectExample(key) {
  try {
    const result = await deleteFromR2(key);
    
    if (result.success) {
      console.log(`Successfully deleted ${key}`);
    } else {
      console.error(`Failed to delete: ${result.error}`);
    }
  } catch (error) {
    console.error('Error in delete example:', error);
  }
}

/**
 * Helper function to detect content type based on file extension
 * @param {string} filePath - Path to the file
 * @returns {string} Content type
 */
function detectContentType(filePath) {
  const extension = filePath.split('.').pop().toLowerCase();
  const contentTypes = {
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'gif': 'image/gif',
    'pdf': 'application/pdf',
    'json': 'application/json',
    'txt': 'text/plain',
    'html': 'text/html',
    'css': 'text/css',
    'js': 'application/javascript'
  };
  
  return contentTypes[extension] || 'application/octet-stream';
}

/**
 * Helper function to format bytes to human-readable format
 * @param {number} bytes - Size in bytes
 * @returns {string} Formatted size
 */
function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Execute examples (uncomment to run)
// uploadFileExample('./data/example.jpg', 'images/example.jpg');
// listObjectsExample('images/');
// downloadFileExample('images/example.jpg', './downloads/example.jpg');
// deleteObjectExample('images/example.jpg'); 