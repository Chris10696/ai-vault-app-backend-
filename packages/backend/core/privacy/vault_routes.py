"""
FastAPI routes for encrypted user vault operations.
Provides endpoints for secure data storage with client-side encryption.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List
import logging

from ..auth.token import decode_token
from .models import VaultCreateRequest, VaultUpdateRequest
from .vault import vault_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vault", tags=["vault"])
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Extract user ID from JWT token."""
    try:
        payload = decode_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing user ID"
            )
        
        return user_id
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed"
        )

@router.post("/entries")
async def create_vault_entry(
    entry_data: VaultCreateRequest,
    vault_password: str = Header(..., alias="X-Vault-Password"),
    user_id: str = Depends(get_current_user)
):
    """
    Create a new encrypted vault entry.
    
    Args:
        entry_data: Entry creation data
        vault_password: User's vault password (from header)
        user_id: Current user ID (from token)
        
    Returns:
        Created entry information
    """
    try:
        if not vault_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vault password is required"
            )
        
        result = await vault_manager.create_user_vault(
            user_id,
            vault_password,
            entry_data
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        return {
            "message": "Vault entry created successfully",
            "entry": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create vault entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create vault entry: {str(e)}"
        )

@router.get("/entries")
async def list_vault_entries(
    user_id: str = Depends(get_current_user)
):
    """
    List all vault entries for the user (metadata only).
    
    Args:
        user_id: Current user ID (from token)
        
    Returns:
        List of vault entry metadata
    """
    try:
        result = await vault_manager.list_vault_entries(user_id)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list vault entries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list vault entries: {str(e)}"
        )

@router.get("/entries/{entry_id}")
async def get_vault_entry(
    entry_id: str,
    vault_password: str = Header(..., alias="X-Vault-Password"),
    user_id: str = Depends(get_current_user)
):
    """
    Retrieve and decrypt a specific vault entry.
    
    Args:
        entry_id: Entry identifier
        vault_password: User's vault password (from header)
        user_id: Current user ID (from token)
        
    Returns:
        Decrypted vault entry
    """
    try:
        if not vault_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vault password is required"
            )
        
        result = await vault_manager.get_vault_entry(
            user_id,
            entry_id,
            vault_password
        )
        
        if not result["success"]:
            if "not found" in result["error"].lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result["error"]
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get vault entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vault entry: {str(e)}"
        )

@router.put("/entries/{entry_id}")
async def update_vault_entry(
    entry_id: str,
    update_data: VaultUpdateRequest,
    vault_password: str = Header(..., alias="X-Vault-Password"),
    user_id: str = Depends(get_current_user)
):
    """
    Update an encrypted vault entry.
    
    Args:
        entry_id: Entry identifier
        update_data: Update request data
        vault_password: User's vault password (from header)
        user_id: Current user ID (from token)
        
    Returns:
        Update confirmation
    """
    try:
        if not vault_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vault password is required"
            )
        
        result = await vault_manager.update_vault_entry(
            user_id,
            entry_id,
            vault_password,
            update_data
        )
        
        if not result["success"]:
            if "not found" in result["error"].lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result["error"]
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update vault entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update vault entry: {str(e)}"
        )

@router.delete("/entries/{entry_id}")
async def delete_vault_entry(
    entry_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Delete a vault entry.
    
    Args:
        entry_id: Entry identifier
        user_id: Current user ID (from token)
        
    Returns:
        Deletion confirmation
    """
    try:
        result = await vault_manager.delete_vault_entry(user_id, entry_id)
        
        if not result["success"]:
            if "not found" in result["error"].lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result["error"]
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete vault entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete vault entry: {str(e)}"
        )

@router.post("/export")
async def export_vault_data(
    vault_password: str = Header(..., alias="X-Vault-Password"),
    user_id: str = Depends(get_current_user)
):
    """
    Export all vault data for the user (decrypted).
    
    Args:
        vault_password: User's vault password (from header)
        user_id: Current user ID (from token)
        
    Returns:
        Exported vault data
    """
    try:
        if not vault_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vault password is required"
            )
        
        result = await vault_manager.export_vault_data(user_id, vault_password)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export vault data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export vault data: {str(e)}"
        )

@router.post("/test-encryption")
async def test_vault_encryption(
    test_data: Dict[str, Any],
    vault_password: str = Header(..., alias="X-Vault-Password")
):
    """
    Test encryption/decryption functionality (for development/testing).
    
    Args:
        test_data: Data to encrypt and decrypt
        vault_password: Password for encryption
        
    Returns:
        Encryption test results
    """
    try:
        from .vault import FernetKeyManager, EncryptedUserVault
        
        if not vault_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vault password is required"
            )
        
        # Generate salt and derive key
        salt = FernetKeyManager.generate_salt()
        key = FernetKeyManager.derive_key_from_password(vault_password, salt)
        
        # Create vault and test encryption
        vault = EncryptedUserVault("test-user", key)
        
        # Encrypt test data
        encrypted = vault.encrypt_data(test_data)
        
        # Decrypt test data
        decrypted = vault.decrypt_data(encrypted)
        
        # Verify data integrity
        data_matches = test_data == decrypted
        
        return {
            "success": True,
            "original_data": test_data,
            "encrypted_size": len(encrypted),
            "decrypted_data": decrypted,
            "data_integrity": data_matches,
            "key_identifier": vault.key_identifier
        }
        
    except Exception as e:
        logger.error(f"Encryption test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Encryption test failed: {str(e)}"
        )

@router.get("/stats")
async def get_vault_statistics(
    user_id: str = Depends(get_current_user)
):
    """
    Get vault usage statistics for the user.
    
    Args:
        user_id: Current user ID (from token)
        
    Returns:
        Vault usage statistics
    """
    try:
        # Get entry list
        entries_result = await vault_manager.list_vault_entries(user_id)
        
        if not entries_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=entries_result["error"]
            )
        
        entries = entries_result["entries"]
        
        # Calculate statistics
        total_entries = len(entries)
        
        # Group by data type
        data_type_counts = {}
        total_access_count = 0
        
        for entry in entries:
            data_type = entry.get("data_type", "unknown")
            data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1
            total_access_count += entry.get("access_count", 0)
        
        # Find most accessed entry
        most_accessed = max(entries, key=lambda x: x.get("access_count", 0)) if entries else None
        
        return {
            "success": True,
            "statistics": {
                "total_entries": total_entries,
                "data_type_distribution": data_type_counts,
                "total_access_count": total_access_count,
                "average_access_per_entry": total_access_count / max(1, total_entries),
                "most_accessed_entry": {
                    "title": most_accessed["title"] if most_accessed else None,
                    "access_count": most_accessed.get("access_count", 0) if most_accessed else 0
                } if most_accessed else None
            },
            "generated_at": vault_manager.supabase.table("encrypted_user_vault").select("created_at").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute().data[0]["created_at"] if vault_manager.supabase else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get vault statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vault statistics: {str(e)}"
        )

@router.get("/health")
async def vault_health_check():
    """Health check endpoint for vault service."""
    return {
        "status": "healthy",
        "service": "vault",
        "encryption": "fernet",
        "key_derivation": "pbkdf2"
    }
