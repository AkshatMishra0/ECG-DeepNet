"""
Google Drive Integration Module
Handles OAuth2 authentication and file upload to Google Drive
"""

import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from io import BytesIO


class GoogleDriveManager:
    """
    Manages Google Drive OAuth2 authentication and file operations
    """
    
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    TOKEN_FILE = 'token.pickle'
    CREDENTIALS_FILE = 'credentials.json'
    
    def __init__(self, credentials_path=None, token_path=None):
        """
        Initialize Google Drive Manager
        
        Args:
            credentials_path: Path to OAuth2 credentials.json file
            token_path: Path to store/load authentication token
        """
        self.credentials_path = credentials_path or self.CREDENTIALS_FILE
        self.token_path = token_path or self.TOKEN_FILE
        self.creds = None
        self.service = None
    
    def is_authenticated(self):
        """Check if user is authenticated"""
        return self.creds is not None and self.creds.valid
    
    def load_credentials(self):
        """Load saved credentials from token file"""
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                self.creds = pickle.load(token)
        
        # Refresh if expired
        if self.creds and self.creds.expired and self.creds.refresh_token:
            try:
                self.creds.refresh(Request())
                self._save_credentials()
            except Exception as e:
                print(f"Failed to refresh credentials: {e}")
                self.creds = None
        
        return self.is_authenticated()
    
    def _save_credentials(self):
        """Save credentials to token file"""
        with open(self.token_path, 'wb') as token:
            pickle.dump(self.creds, token)
    
    def get_authorization_url(self, redirect_uri):
        """
        Generate OAuth2 authorization URL
        
        Args:
            redirect_uri: OAuth redirect URI configured in Google Cloud Console
            
        Returns:
            Tuple of (authorization_url, state)
        """
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found: {self.credentials_path}\n"
                "Please download credentials.json from Google Cloud Console"
            )
        
        flow = Flow.from_client_secrets_file(
            self.credentials_path,
            scopes=self.SCOPES,
            redirect_uri=redirect_uri
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        return authorization_url, state
    
    def authenticate_from_code(self, code, redirect_uri, state=None):
        """
        Exchange authorization code for credentials
        
        Args:
            code: Authorization code from OAuth callback
            redirect_uri: OAuth redirect URI
            state: OAuth state parameter (optional)
            
        Returns:
            bool: True if authentication successful
        """
        try:
            flow = Flow.from_client_secrets_file(
                self.credentials_path,
                scopes=self.SCOPES,
                redirect_uri=redirect_uri,
                state=state
            )
            
            flow.fetch_token(code=code)
            self.creds = flow.credentials
            self._save_credentials()
            return True
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False
    
    def revoke_authentication(self):
        """Revoke current authentication"""
        if os.path.exists(self.token_path):
            os.remove(self.token_path)
        self.creds = None
        self.service = None
    
    def _get_service(self):
        """Get or create Drive API service"""
        if not self.service:
            if not self.is_authenticated():
                raise ValueError("Not authenticated. Please authenticate first.")
            self.service = build('drive', 'v3', credentials=self.creds)
        return self.service
    
    def upload_file(self, file_content, filename, mimetype='application/pdf', folder_id=None):
        """
        Upload a file to Google Drive
        
        Args:
            file_content: BytesIO or file-like object containing file data
            filename: Name for the file in Google Drive
            mimetype: MIME type of the file
            folder_id: Optional folder ID to upload to
            
        Returns:
            dict: File metadata from Google Drive API
        """
        try:
            service = self._get_service()
            
            file_metadata = {'name': filename}
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            # Ensure file_content is at the beginning
            if hasattr(file_content, 'seek'):
                file_content.seek(0)
            
            media = MediaIoBaseUpload(
                file_content,
                mimetype=mimetype,
                resumable=True
            )
            
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink'
            ).execute()
            
            return {
                'success': True,
                'file_id': file.get('id'),
                'name': file.get('name'),
                'link': file.get('webViewLink')
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_folder(self, folder_name, parent_folder_id=None):
        """
        Create a folder in Google Drive
        
        Args:
            folder_name: Name of the folder to create
            parent_folder_id: Optional parent folder ID
            
        Returns:
            str: Folder ID if successful, None otherwise
        """
        try:
            service = self._get_service()
            
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_folder_id:
                file_metadata['parents'] = [parent_folder_id]
            
            folder = service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            return folder.get('id')
        except Exception as e:
            print(f"Failed to create folder: {e}")
            return None
    
    def list_files(self, page_size=10, folder_id=None):
        """
        List files in Google Drive
        
        Args:
            page_size: Number of files to return
            folder_id: Optional folder ID to list files from
            
        Returns:
            list: List of file metadata dictionaries
        """
        try:
            service = self._get_service()
            
            query = ""
            if folder_id:
                query = f"'{folder_id}' in parents"
            
            results = service.files().list(
                pageSize=page_size,
                fields="files(id, name, mimeType, createdTime, webViewLink)",
                q=query
            ).execute()
            
            return results.get('files', [])
        except Exception as e:
            print(f"Failed to list files: {e}")
            return []
