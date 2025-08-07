"""
Encrypted user vault implementation using Fernet symmetric encryption.
Provides secure client-side encryption for sensitive user data storage.
"""

import base64
import json
import logging
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from supabase import create_client, Client

from .models import VaultEntry, VaultCreateRequest, VaultUpdateRequest

logger = logging.getLogger(__name__)

class FernetKeyManager:
    """
    Manages Fernet encryption keys with password-based key derivation.
    Provides secure key generation and derivation from user passwords.
    """
    
    @staticmethod
    def generate_key() -> bytes:
        """
        Generate a new Fernet key.
        
        Returns:
            32-byte Fernet key
        """
        return Fernet.generate_key()
    
    @staticmethod
    def derive_key_from_password(password: str, salt: bytes) -> bytes:
        """
        Derive a Fernet key from a password using PBKDF2.
        
        Args:
            password: User password
            salt: Random salt (16 bytes recommended)
            
        Returns:
            Derived Fernet key
        """
        if isinstance(password, str):
            password = password.encode('utf-8')
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # Fernet requires 32-byte keys
            salt=salt,
            iterations=600_000,  # OWASP recommended minimum for 2023
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    @staticmethod
    def generate_salt() -> bytes:
        """
        Generate a random salt for key derivation.
        
        Returns:
            16-byte random salt
        """
        return os.urandom(16)
    
    @staticmethod
    def hash_key_identifier(key: bytes) -> str:
        """
        Create a hash identifier for a key (for key verification).
        
        Args:
            key: Fernet key
            
        Returns:
            SHA-256 hash of the key (hex encoded)
        """
        return hashlib.sha256(key).hexdigest()[:16]  # First 16 chars

class EncryptedUserVault:
    """
    Client-side encrypted user vault for secure data storage.
    Uses Fernet symmetric encryption with password-derived keys.
    """
    
    def __init__(self, user_id: str, encryption_key: bytes):
        """
        Initialize encrypted vault for a user.
        
        Args:
            user_id: User identifier
            encryption_key: Fernet encryption key
        """
        self.user_id = user_id
        self.fernet = Fernet(encryption_key)
        self.key_identifier = FernetKeyManager.hash_key_identifier(encryption_key)
        logger.debug(f"Initialized vault for user {user_id}")
    
    def encrypt_data(self, data: Dict[str, Any]) -> str:
        """
        Encrypt data using Fernet encryption.
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            Base64 encoded encrypted data
        """
        try:
            # Convert data to JSON bytes
            json_data = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            data_bytes = json_data.encode('utf-8')
            
            # Encrypt data
            encrypted_bytes = self.fernet.encrypt(data_bytes)
            
            # Return base64 encoded encrypted data
            encrypted_b64 = base64.b64encode(encrypted_bytes).decode('ascii')
            
            logger.debug(f"Encrypted data for user {self.user_id} (size: {len(data_bytes)} bytes)")
            return encrypted_b64
            
        except Exception as e:
            logger.error(f"Encryption failed for user {self.user_id}: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt data using Fernet decryption.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted dictionary
        """
        try:
            # Decode base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode('ascii'))
            
            # Decrypt data
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            
            # Convert back to dictionary
            json_data = decrypted_bytes.decode('utf-8')
            data = json.loads(json_data)
            
            logger.debug(f"Decrypted data for user {self.user_id}")
            return data
            
        except Exception as e:
            logger.error(f"Decryption failed for user {self.user_id}: {e}")
            raise
    
    def encrypt_vault_entry(self, entry_data: VaultCreateRequest) -> VaultEntry:
        """
        Create an encrypted vault entry.
        
        Args:
            entry_data: Entry creation request
            
        Returns:
            VaultEntry with encrypted data
        """
        try:
            # Prepare data for encryption
            data_to_encrypt = {
                "data": entry_data.data,
                "encrypted_at": datetime.utcnow().isoformat(),
                "key_identifier": self.key_identifier
            }
            
            # Encrypt the data
            encrypted_data = self.encrypt_data(data_to_encrypt)
            
            # Create vault entry
            vault_entry = VaultEntry(
                user_id=self.user_id,
                title=entry_data.title,
                encrypted_data=encrypted_data,
                data_type=entry_data.data_type,
                tags=entry_data.tags
            )
            
            logger.info(f"Created encrypted vault entry for user {self.user_id}")
            return vault_entry
            
        except Exception as e:
            logger.error(f"Failed to create vault entry: {e}")
            raise
    
    def decrypt_vault_entry(self, vault_entry: VaultEntry) -> Dict[str, Any]:
        """
        Decrypt a vault entry.
        
        Args:
            vault_entry: Encrypted vault entry
            
        Returns:
            Decrypted data dictionary
        """
        try:
            # Decrypt the data
            decrypted_container = self.decrypt_data(vault_entry.encrypted_data)
            
            # Verify key identifier
            stored_key_id = decrypted_container.get("key_identifier")
            if stored_key_id != self.key_identifier:
                logger.warning(f"Key identifier mismatch for entry {vault_entry.entry_id}")
                # Continue anyway - key might have been rotated
            
            # Return the actual data
            return decrypted_container.get("data", {})
            
        except Exception as e:
            logger.error(f"Failed to decrypt vault entry {vault_entry.entry_id}: {e}")
            raise
    
    def update_vault_entry(
        self, 
        vault_entry: VaultEntry, 
        update_data: VaultUpdateRequest
    ) -> VaultEntry:
        """
        Update an encrypted vault entry.
        
        Args:
            vault_entry: Existing vault entry
            update_data: Update request
            
        Returns:
            Updated vault entry
        """
        try:
            # If data is being updated, decrypt current data first
            if update_data.data is not None:
                # Prepare new data for encryption
                data_to_encrypt = {
                    "data": update_data.data,
                    "encrypted_at": datetime.utcnow().isoformat(),
                    "key_identifier": self.key_identifier,
                    "updated_from": vault_entry.entry_id
                }
                
                # Encrypt the new data
                vault_entry.encrypted_data = self.encrypt_data(data_to_encrypt)
            
            # Update other fields
            if update_data.title is not None:
                vault_entry.title = update_data.title
            
            if update_data.tags is not None:
                vault_entry.tags = update_data.tags
            
            vault_entry.updated_at = datetime.utcnow()
            
            logger.info(f"Updated vault entry {vault_entry.entry_id}")
            return vault_entry
            
        except Exception as e:
            logger.error(f"Failed to update vault entry {vault_entry.entry_id}: {e}")
            raise

class VaultManager:
    """
    High-level vault management with database integration.
    Handles vault operations with secure key management and persistence.
    """
    
    def __init__(self):
        self.supabase: Optional[Client] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Supabase client."""
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            if supabase_url and supabase_key:
                self.supabase = create_client(supabase_url, supabase_key)
            logger.info("VaultManager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize VaultManager: {e}")
    
    async def create_user_vault(
        self, 
        user_id: str, 
        password: str,
        entry_data: VaultCreateRequest
    ) -> Dict[str, Any]:
        """
        Create a new vault entry for a user with password-based encryption.
        
        Args:
            user_id: User identifier
            password: User's vault password
            entry_data: Entry creation request
            
        Returns:
            Success response with entry details
        """
        if not self.supabase:
            raise RuntimeError("Supabase client not initialized")
        
        try:
            # Generate salt and derive key
            salt = FernetKeyManager.generate_salt()
            encryption_key = FernetKeyManager.derive_key_from_password(password, salt)
            
            # Create encrypted vault
            vault = EncryptedUserVault(user_id, encryption_key)
            
            # Create encrypted entry
            vault_entry = vault.encrypt_vault_entry(entry_data)
            
            # Store in database
            entry_dict = vault_entry.dict()
            # Convert datetime objects to ISO strings
            for key, value in entry_dict.items():
                if isinstance(value, datetime):
                    entry_dict[key] = value.isoformat()
            
            # Store salt with the entry for key derivation
            entry_dict['key_salt'] = base64.b64encode(salt).decode('ascii')
            
            result = self.supabase.table("encrypted_user_vault").insert(entry_dict).execute()
            
            if not result.data:
                raise Exception("Failed to store vault entry")
            
            stored_entry = result.data[0]
            
            logger.info(f"Created vault entry {stored_entry['entry_id']} for user {user_id}")
            
            return {
                "success": True,
                "entry_id": stored_entry['entry_id'],
                "title": stored_entry['title'],
                "data_type": stored_entry['data_type'],
                "tags": stored_entry['tags'],
                "created_at": stored_entry['created_at']
            }
            
        except Exception as e:
            logger.error(f"Failed to create vault entry for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_vault_entry(
        self, 
        user_id: str, 
        entry_id: str, 
        password: str
    ) -> Dict[str, Any]:
        """
        Retrieve and decrypt a vault entry.
        
        Args:
            user_id: User identifier
            entry_id: Entry identifier
            password: User's vault password
            
        Returns:
            Decrypted vault entry or error
        """
        if not self.supabase:
            raise RuntimeError("Supabase client not initialized")
        
        try:
            # Get entry from database
            result = self.supabase.table("encrypted_user_vault").select("*").eq(
                "entry_id", entry_id
            ).eq("user_id", user_id).execute()
            
            if not result.data:
                return {
                    "success": False,
                    "error": "Vault entry not found"
                }
            
            entry_data = result.data[0]
            
            # Decode salt and derive key
            salt = base64.b64decode(entry_data['key_salt'].encode('ascii'))
            encryption_key = FernetKeyManager.derive_key_from_password(password, salt)
            
            # Create vault and decrypt entry
            vault = EncryptedUserVault(user_id, encryption_key)
            vault_entry = VaultEntry(**entry_data)
            
            decrypted_data = vault.decrypt_vault_entry(vault_entry)
            
            # Update access count
            new_access_count = entry_data['access_count'] + 1
            self.supabase.table("encrypted_user_vault").update({
                "access_count": new_access_count
            }).eq("entry_id", entry_id).execute()
            
            logger.info(f"Retrieved vault entry {entry_id} for user {user_id}")
            
            return {
                "success": True,
                "entry_id": entry_id,
                "title": entry_data['title'],
                "data": decrypted_data,
                "data_type": entry_data['data_type'],
                "tags": entry_data['tags'],
                "created_at": entry_data['created_at'],
                "updated_at": entry_data['updated_at'],
                "access_count": new_access_count
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve vault entry {entry_id} for user {user_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to decrypt vault entry: {str(e)}"
            }
    
    async def list_vault_entries(self, user_id: str) -> Dict[str, Any]:
        """
        List all vault entries for a user (metadata only, no decryption).
        
        Args:
            user_id: User identifier
            
        Returns:
            List of vault entry metadata
        """
        if not self.supabase:
            raise RuntimeError("Supabase client not initialized")
        
        try:
            result = self.supabase.table("encrypted_user_vault").select(
                "entry_id, title, data_type, tags, created_at, updated_at, access_count"
            ).eq("user_id", user_id).order("created_at", desc=True).execute()
            
            entries = result.data if result.data else []
            
            logger.info(f"Listed {len(entries)} vault entries for user {user_id}")
            
            return {
                "success": True,
                "entries": entries,
                "total_count": len(entries)
            }
            
        except Exception as e:
            logger.error(f"Failed to list vault entries for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_vault_entry(
        self, 
        user_id: str, 
        entry_id: str, 
        password: str,
        update_data: VaultUpdateRequest
    ) -> Dict[str, Any]:
        """
        Update a vault entry.
        
        Args:
            user_id: User identifier
            entry_id: Entry identifier
            password: User's vault password
            update_data: Update request
            
        Returns:
            Update result
        """
        if not self.supabase:
            raise RuntimeError("Supabase client not initialized")
        
        try:
            # Get existing entry
            result = self.supabase.table("encrypted_user_vault").select("*").eq(
                "entry_id", entry_id
            ).eq("user_id", user_id).execute()
            
            if not result.data:
                return {
                    "success": False,
                    "error": "Vault entry not found"
                }
            
            entry_data = result.data[0]
            
            # Decode salt and derive key
            salt = base64.b64decode(entry_data['key_salt'].encode('ascii'))
            encryption_key = FernetKeyManager.derive_key_from_password(password, salt)
            
            # Create vault and update entry
            vault = EncryptedUserVault(user_id, encryption_key)
            vault_entry = VaultEntry(**entry_data)
            
            updated_entry = vault.update_vault_entry(vault_entry, update_data)
            
            # Prepare update data
            update_dict = {
                "title": updated_entry.title,
                "encrypted_data": updated_entry.encrypted_data,
                "tags": updated_entry.tags,
                "updated_at": updated_entry.updated_at.isoformat()
            }
            
            # Update in database
            self.supabase.table("encrypted_user_vault").update(update_dict).eq(
                "entry_id", entry_id
            ).execute()
            
            logger.info(f"Updated vault entry {entry_id} for user {user_id}")
            
            return {
                "success": True,
                "entry_id": entry_id,
                "message": "Vault entry updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to update vault entry {entry_id} for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_vault_entry(
        self, 
        user_id: str, 
        entry_id: str
    ) -> Dict[str, Any]:
        """
        Delete a vault entry.
        
        Args:
            user_id: User identifier
            entry_id: Entry identifier
            
        Returns:
            Deletion result
        """
        if not self.supabase:
            raise RuntimeError("Supabase client not initialized")
        
        try:
            # Delete entry
            result = self.supabase.table("encrypted_user_vault").delete().eq(
                "entry_id", entry_id
            ).eq("user_id", user_id).execute()
            
            if not result.data:
                return {
                    "success": False,
                    "error": "Vault entry not found or already deleted"
                }
            
            logger.info(f"Deleted vault entry {entry_id} for user {user_id}")
            
            return {
                "success": True,
                "message": "Vault entry deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete vault entry {entry_id} for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def export_vault_data(
        self, 
        user_id: str, 
        password: str
    ) -> Dict[str, Any]:
        """
        Export all vault data for a user (decrypted).
        
        Args:
            user_id: User identifier
            password: User's vault password
            
        Returns:
            Exported vault data
        """
        try:
            # Get all entries
            entries_result = await self.list_vault_entries(user_id)
            if not entries_result["success"]:
                return entries_result
            
            exported_entries = []
            
            # Decrypt each entry
            for entry_meta in entries_result["entries"]:
                entry_result = await self.get_vault_entry(
                    user_id, 
                    entry_meta["entry_id"], 
                    password
                )
                
                if entry_result["success"]:
                    exported_entries.append({
                        "title": entry_result["title"],
                        "data": entry_result["data"],
                        "data_type": entry_result["data_type"],
                        "tags": entry_result["tags"],
                        "created_at": entry_result["created_at"],
                        "updated_at": entry_result["updated_at"]
                    })
            
            logger.info(f"Exported {len(exported_entries)} vault entries for user {user_id}")
            
            return {
                "success": True,
                "entries": exported_entries,
                "exported_at": datetime.utcnow().isoformat(),
                "total_entries": len(exported_entries)
            }
            
        except Exception as e:
            logger.error(f"Failed to export vault data for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global vault manager instance
vault_manager = VaultManager()
