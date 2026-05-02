"""Helpers navigation multipages (query params)."""

import streamlit as st


def get_query_param(key: str, default=None):
    """
    Lit un paramètre d'URL Streamlit (st.query_params).
    Retourne une chaîne unique ou default.
    """
    try:
        if key not in st.query_params:
            return default
        v = st.query_params[key]
        if v is None:
            return default
        if isinstance(v, (list, tuple)):
            return v[0] if len(v) > 0 else default
        return str(v) if v != "" else default
    except Exception:
        return default
