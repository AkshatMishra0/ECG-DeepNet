# Google Drive Integration Setup Guide

This guide will help you set up Google Drive integration for ECG-DeepNet to upload PDF reports directly to Google Drive.

## Prerequisites

- A Google Account
- Access to Google Cloud Console

## Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top and select **"New Project"**
3. Enter a project name (e.g., "ECG-DeepNet") and click **"Create"**
4. Wait for the project to be created and select it

## Step 2: Enable Google Drive API

1. In your Google Cloud project, go to **"APIs & Services" > "Library"**
2. Search for **"Google Drive API"**
3. Click on it and then click **"Enable"**

## Step 3: Configure OAuth Consent Screen

1. Go to **"APIs & Services" > "OAuth consent screen"**
2. Select **"External"** user type (unless you have a Google Workspace account)
3. Click **"Create"**
4. Fill in the required fields:
   - **App name**: ECG-DeepNet
   - **User support email**: Your email
   - **Developer contact information**: Your email
5. Click **"Save and Continue"**
6. On the **Scopes** page, click **"Add or Remove Scopes"**
7. Search for **"Google Drive API"** and select:
   - `.../auth/drive.file` (View and manage Google Drive files created by this app)
8. Click **"Update"** and then **"Save and Continue"**
9. On **Test users**, add your Google account email
10. Click **"Save and Continue"**

## Step 4: Create OAuth 2.0 Credentials

1. Go to **"APIs & Services" > "Credentials"**
2. Click **"Create Credentials" > "OAuth client ID"**
3. Select **"Web application"** as the application type
4. Enter a name: "ECG-DeepNet Web Client"
5. Under **"Authorized redirect URIs"**, add:
   ```
   http://localhost:10000/google_drive/callback
   ```
   (If deploying to production, also add your production URL with `/google_drive/callback`)
6. Click **"Create"**
7. Download the credentials JSON file
8. **Important**: Rename the downloaded file to `credentials.json`

## Step 5: Add Credentials to Your Project

1. Copy the `credentials.json` file to your ECG-DeepNet project root directory:
   ```
   ECG-DeepNet/
   ├── credentials.json    ← Place the file here
   ├── flask_app.py
   ├── requirements.txt
   └── ...
   ```

2. **Security Warning**: Add `credentials.json` and `token.pickle` to your `.gitignore`:
   ```bash
   echo credentials.json >> .gitignore
   echo token.pickle >> .gitignore
   ```

## Step 6: Install Dependencies

Install the required Google API packages:

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Step 7: Test the Integration

1. Start your Flask application:
   ```bash
   python flask_app.py
   ```

2. Open your browser and go to `http://localhost:10000`

3. Run an ECG prediction to generate a report

4. On the report page, click **"Upload to Google Drive"**

5. If prompted, click **"Connect Google Drive"**

6. You'll be redirected to Google's authorization page:
   - Select your Google account
   - Review the permissions
   - Click **"Continue"** (you may see a warning about unverified app - this is normal for testing)
   - Grant the requested permissions

7. You'll be redirected back to the app

8. Click **"Upload to Google Drive"** again

9. The PDF report should now be uploaded to your Google Drive!

## Verify Upload

1. Go to [Google Drive](https://drive.google.com/)
2. You should see your ECG report PDF in the "My Drive" section
3. The file name will be in format: `ecg_report_YYYYMMDD_HHMMSS.pdf` or `PatientID_YYYYMMDD_HHMMSS.pdf`

## Troubleshooting

### "credentials.json not found" Error

- Make sure the `credentials.json` file is in the project root directory
- Verify the filename is exactly `credentials.json` (not `client_secret_*.json`)

### "Not authenticated with Google Drive" Error

- Click the "Connect Google Drive" button on the report page
- Complete the OAuth authorization flow
- Try uploading again

### "Authorization failed" Error

- Check that the redirect URI in Google Cloud Console matches your Flask app URL
- For local development: `http://localhost:10000/google_drive/callback`
- Make sure the Google Drive API is enabled in your project

### "This app hasn't been verified" Warning

This is normal during development. To continue:
1. Click **"Advanced"**
2. Click **"Go to ECG-DeepNet (unsafe)"**
3. Grant permissions

To remove this warning in production, submit your app for verification in Google Cloud Console.

## Production Deployment

When deploying to production:

1. Update the authorized redirect URIs in Google Cloud Console:
   ```
   https://yourdomain.com/google_drive/callback
   ```

2. Update the OAuth consent screen to "Published" status (requires verification for production use)

3. Use environment variables for sensitive data:
   ```python
   CREDENTIALS_PATH = os.environ.get('GOOGLE_CREDENTIALS_PATH', 'credentials.json')
   ```

4. **Never commit** `credentials.json` or `token.pickle` to version control

## Revoking Access

To revoke Google Drive access:

1. In the app, there's a revoke endpoint (can add a button if needed)
2. Or go to [Google Account > Security > Third-party apps](https://myaccount.google.com/permissions)
3. Find "ECG-DeepNet" and click **"Remove Access"**

## Additional Features

The `GoogleDriveManager` class supports:

- **Creating folders**: Organize reports by date or patient
- **Listing files**: View uploaded reports
- **Batch uploads**: Upload multiple files at once

See `utils/google_drive.py` for available methods.

## Support

For issues:
1. Check the Flask app logs for detailed error messages
2. Verify all steps in this guide were completed
3. Ensure Google Drive API quota hasn't been exceeded (check Google Cloud Console)

---

**Important Security Notes:**
- Keep `credentials.json` secure and never share it
- Add `credentials.json` and `token.pickle` to `.gitignore`
- The token file (`token.pickle`) stores user authentication - protect it
- Use environment variables for production deployments
