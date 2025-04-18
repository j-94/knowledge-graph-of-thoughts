# Cloudflare R2 Integration

This module provides an easy way to integrate Cloudflare R2 storage with the Knowledge Graph of Thoughts project. It allows for uploading, downloading, listing, and deleting objects from an R2 bucket.

## Features

- Upload files to Cloudflare R2
- Download files from Cloudflare R2
- List objects in your R2 bucket
- Delete objects from your R2 bucket

## Setup

### Prerequisites

1. A Cloudflare account with R2 enabled
2. An R2 bucket created in your Cloudflare account
3. API access keys for your R2 bucket

### Environment Variables

Create a `.env` file with the following variables (see `.env.example` in the root directory):

```
R2_ACCESS_KEY_ID=your_access_key_id
R2_SECRET_ACCESS_KEY=your_secret_access_key
R2_BUCKET_NAME=your_bucket_name
R2_ENDPOINT=https://your_account_id.r2.cloudflarestorage.com
```

### Installation

The necessary dependencies are already included in the `package.json`:

```bash
npm install
```

## Usage

This module exports several functions for interacting with your R2 bucket:

### Upload a File

```javascript
import { uploadToR2 } from './r2-operations.js';
import fs from 'fs';

// Read a local file
const fileContent = fs.readFileSync('./path/to/file.jpg');

// Upload to R2
const result = await uploadToR2('images/file.jpg', fileContent, 'image/jpeg');
console.log(result);
```

### Download a File

```javascript
import { getFromR2 } from './r2-operations.js';
import fs from 'fs';

// Get from R2
const result = await getFromR2('images/file.jpg');

// Save to local file system
if (result.success) {
  const chunks = [];
  for await (const chunk of result.data) {
    chunks.push(chunk);
  }
  fs.writeFileSync('./downloads/file.jpg', Buffer.concat(chunks));
}
```

### List Objects

```javascript
import { listR2Objects } from './r2-operations.js';

// List all objects
const allObjects = await listR2Objects();
console.log(allObjects.objects);

// List objects with prefix (like a directory)
const imagesOnly = await listR2Objects('images/');
console.log(imagesOnly.objects);
```

### Delete an Object

```javascript
import { deleteFromR2 } from './r2-operations.js';

// Delete an object
const result = await deleteFromR2('images/file.jpg');
console.log(result);
```

## Example Implementation

See `r2-example.js` for complete usage examples. 