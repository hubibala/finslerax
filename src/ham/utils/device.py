"""Device configuration utilities for HAMTools.

Provides a thin wrapper around JAX's device selection mechanism, allowing
experiments and the training pipeline to target CPU or GPU with a single
string argument.

Usage::

    from ham.utils import configure_device, get_device

    # At the top of a script, before creating any arrays:
    dev = configure_device("gpu")   # or "cpu" (default)

    # Or retrieve a device object without changing the global default:
    dev = get_device("gpu")
    data = jax.device_put(data, dev)

Alternatively, set the ``JAX_PLATFORMS`` environment variable before
importing JAX::

    JAX_PLATFORMS=gpu python examples/experiment_gahtan_phase1.py
"""

import jax

__all__ = ["get_device", "configure_device"]


def get_device(device: str = "cpu") -> jax.Device:
    """Return the first available JAX device matching *device*.

    Args:
        device: ``"cpu"``, ``"gpu"``, or ``"tpu"``.

    Returns:
        The first ``jax.Device`` of the requested type.

    Raises:
        RuntimeError: If the requested backend has no available devices.
    """
    try:
        return jax.devices(device)[0]
    except RuntimeError as exc:
        available = [str(d) for d in jax.devices()]
        raise RuntimeError(
            f"Device '{device}' not available. "
            f"Available devices: {available}"
        ) from exc


def configure_device(device: str) -> jax.Device:
    """Set the global JAX default device and return it.

    Call this once at script startup, after importing JAX but before creating
    any arrays. All subsequent JAX operations (including JIT-compiled functions
    and ``jax.device_put`` calls inside the training pipeline) will target the
    chosen device.

    Internally calls ``jax.config.update("jax_default_device", ...)``.

    Args:
        device: ``"cpu"``, ``"gpu"``, or ``"tpu"``.

    Returns:
        The selected ``jax.Device``.

    Example:
        >>> from ham.utils import configure_device
        >>> dev = configure_device("gpu")   # all subsequent JAX ops → GPU
        >>> dev = configure_device("cpu")   # fall back to CPU
    """
    dev = get_device(device)
    jax.config.update("jax_default_device", dev)
    print(f"[HAMTools] Default JAX device set to: {dev}")
    return dev
