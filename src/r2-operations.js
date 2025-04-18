import { 
  PutObjectCommand, 
  GetObjectCommand, 
  ListObjectsV2Command, 
  DeleteObjectCommand 
} from "@aws-sdk/client-s3";
import { r2Client, BUCKET_NAME } from "./r2-client.js";

/**
 * Upload a file or data to Cloudflare R2
 * @param {string} key - Object key/path in the bucket
 * @param {Buffer|Blob|string|ReadableStream} data - The data to upload
 * @param {string} contentType - MIME type of the content
 * @returns {Promise<Object>} Upload result
 */
export async function uploadToR2(key, data, contentType = "application/octet-stream") {
  try {
    const command = new PutObjectCommand({
      Bucket: BUCKET_NAME,
      Key: key,
      Body: data,
      ContentType: contentType,
    });
    
    const response = await r2Client.send(command);
    return { 
      success: true, 
      key, 
      etag: response.ETag 
    };
  } catch (error) {
    console.error("Error uploading to R2:", error);
    return { success: false, error: error.message };
  }
}

/**
 * Download a file from Cloudflare R2
 * @param {string} key - Object key/path in the bucket
 * @returns {Promise<Object>} Download result with data stream
 */
export async function getFromR2(key) {
  try {
    const command = new GetObjectCommand({
      Bucket: BUCKET_NAME,
      Key: key,
    });
    
    const response = await r2Client.send(command);
    return {
      success: true,
      data: response.Body,
      contentType: response.ContentType,
      metadata: response.Metadata
    };
  } catch (error) {
    console.error("Error downloading from R2:", error);
    return { success: false, error: error.message };
  }
}

/**
 * List objects in the R2 bucket
 * @param {string} prefix - Optional prefix to filter objects
 * @returns {Promise<Object>} List of objects
 */
export async function listR2Objects(prefix = "") {
  try {
    const command = new ListObjectsV2Command({
      Bucket: BUCKET_NAME,
      Prefix: prefix,
    });
    
    const response = await r2Client.send(command);
    return {
      success: true,
      objects: response.Contents || []
    };
  } catch (error) {
    console.error("Error listing R2 objects:", error);
    return { success: false, error: error.message };
  }
}

/**
 * Delete an object from the R2 bucket
 * @param {string} key - Object key/path to delete
 * @returns {Promise<Object>} Delete operation result
 */
export async function deleteFromR2(key) {
  try {
    const command = new DeleteObjectCommand({
      Bucket: BUCKET_NAME,
      Key: key,
    });
    
    await r2Client.send(command);
    return { success: true, key };
  } catch (error) {
    console.error("Error deleting from R2:", error);
    return { success: false, error: error.message };
  }
} 