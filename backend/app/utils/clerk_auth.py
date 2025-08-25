"""Clerk authentication utilities for JWT token validation."""

import logging
from typing import Any

import httpx
from clerk_backend_api import Clerk
from clerk_backend_api.security.types import AuthenticateRequestOptions
from fastapi import HTTPException, status

from app.core.settings import settings

logger = logging.getLogger(__name__)


async def verify_clerk_token(token: str) -> dict[str, Any]:
    """
    Verify a Clerk JWT token and return user information.

    Args:
        token: The JWT token from the Authorization header

    Returns:
        Dict containing user information from Clerk

    Raises:
        HTTPException: If token is invalid or user not found
    """
    if not settings.CLERK_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CLERK_SECRET_KEY is not configured",
        )

    try:
        # Create Clerk SDK instance
        clerk = Clerk(bearer_auth=settings.CLERK_SECRET_KEY)

        # Create a proper httpx.Request object with the token
        request = httpx.Request(
            method="GET",
            url="https://api.clerk.com/",  # Dummy URL, not used for authentication
            headers={"Authorization": f"Bearer {token}"},
        )

        # Authenticate the request using Clerk SDK
        request_state = clerk.authenticate_request(
            request, AuthenticateRequestOptions()
        )

        # Check if the request is authenticated
        if not request_state.is_signed_in:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Get user ID from the token payload
        user_id = request_state.payload.get("sub") if request_state.payload else None

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Try to get user email from Clerk API
        try:
            user_data = clerk.users.get(user_id=user_id)

            # Extract primary email address
            email = None
            if user_data and user_data.email_addresses:
                for email_addr in user_data.email_addresses:
                    if hasattr(email_addr, "primary") and email_addr.primary:
                        email = email_addr.email_address
                        break

                # Fallback to first email if no primary found
                if not email and user_data.email_addresses:
                    email = user_data.email_addresses[0].email_address

        except Exception as e:
            logger.warning(f"Could not fetch user details from Clerk: {str(e)}")
            # Fallback to email from token payload
            email = (
                request_state.payload.get("email") if request_state.payload else None
            )

        return {
            "id": user_id,
            "email": email,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error verifying Clerk token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
