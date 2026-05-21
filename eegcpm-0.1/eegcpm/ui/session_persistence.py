"""Session state persistence via browser localStorage.

Saves and restores core project session state across page refreshes.
Only persists keys that are safe to store as plain strings.
"""

import streamlit as st
from streamlit_local_storage import LocalStorage

# Keys to persist across refreshes (all string-serialisable)
_PERSISTENT_KEYS = ["current_project_name", "bids_root", "eegcpm_root"]

# prefix to avoid collisions with other apps
_LS_NAMESPACE = "eegcpm_"


def _get_storage() -> LocalStorage:
    """Return a cached LocalStorage instance (one per session)."""
    if "_local_storage" not in st.session_state:
        st.session_state._local_storage = LocalStorage(key="eegcpm_persistence")
    return st.session_state._local_storage


def save_project_to_storage() -> None:
    """Write current project session state keys to localStorage."""
    ls = _get_storage()
    for i, key in enumerate(_PERSISTENT_KEYS):
        value = st.session_state.get(key)
        if value is not None:
            ls.setItem(
                itemKey=f"{_LS_NAMESPACE}{key}",
                itemValue=str(value),
                key=f"save_{key}_{i}",
            )


def restore_project_from_storage() -> bool:
    """Restore project session state from localStorage if session state is missing.

    Returns True if any values were restored (caller may want to rerun).
    """
    # Only restore keys that are currently missing from session_state
    missing = [k for k in _PERSISTENT_KEYS if k not in st.session_state]
    if not missing:
        return False

    ls = _get_storage()
    restored = False
    for key in missing:
        value = ls.getItem(f"{_LS_NAMESPACE}{key}")
        if value:
            st.session_state[key] = value
            restored = True

    # Also restore ProjectManager if we got a project name
    if restored and "current_project_name" in st.session_state:
        if "project_manager" not in st.session_state:
            from eegcpm.ui.project_manager import ProjectManager
            st.session_state.project_manager = ProjectManager()

    return restored


def clear_project_from_storage() -> None:
    """Remove project keys from localStorage (e.g. on project delete)."""
    ls = _get_storage()
    for key in _PERSISTENT_KEYS:
        ls.deleteItem(itemKey=f"{_LS_NAMESPACE}{key}")
