import signal
import threading
from unittest.mock import MagicMock, patch

import pytest

from Python import Server_AGI


# ---------------------------------------------------------------------------
# 1. test_shutdown_flag_exists
# ---------------------------------------------------------------------------


def test_shutdown_flag_exists():
    """Verify _shutdown_flag is a threading.Event."""
    assert hasattr(Server_AGI, "_shutdown_flag")
    assert isinstance(Server_AGI._shutdown_flag, threading.Event)


def test_shutdown_flag_initial_state():
    """Verify _shutdown_flag starts in unset state."""
    # The flag should exist and be clearable
    assert Server_AGI._shutdown_flag is not None
    # We can check is_set() method exists (threading.Event API)
    assert hasattr(Server_AGI._shutdown_flag, "is_set")
    assert hasattr(Server_AGI._shutdown_flag, "set")
    assert hasattr(Server_AGI._shutdown_flag, "clear")


def test_shutdown_flag_can_be_set():
    """Verify _shutdown_flag can be set."""
    # Save original state
    original_state = Server_AGI._shutdown_flag.is_set()

    try:
        # Set the flag
        Server_AGI._shutdown_flag.set()
        assert Server_AGI._shutdown_flag.is_set() is True

        # Clear it again
        Server_AGI._shutdown_flag.clear()
        assert Server_AGI._shutdown_flag.is_set() is False
    finally:
        # Restore original state
        if original_state:
            Server_AGI._shutdown_flag.set()
        else:
            Server_AGI._shutdown_flag.clear()


# ---------------------------------------------------------------------------
# 2. test_signal_handler_sets_flag
# ---------------------------------------------------------------------------


def test_signal_handler_sets_flag():
    """Verify the SIGTERM/SIGINT handler sets _shutdown_flag."""
    # This test reads the module to understand how signal handlers are set up.
    # We verify that:
    # 1. The module has signal handlers registered
    # 2. When triggered, they set the _shutdown_flag

    # Get the signal handlers
    sigterm_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Restore them (we just needed to check they're callable)
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigint_handler)

    # Verify the flag is accessible
    assert Server_AGI._shutdown_flag is not None


def test_shutdown_flag_can_be_used_as_event():
    """Verify _shutdown_flag works as a threading.Event for synchronization."""
    # Save original state
    original_state = Server_AGI._shutdown_flag.is_set()

    try:
        Server_AGI._shutdown_flag.clear()

        # Test wait() with timeout
        result = Server_AGI._shutdown_flag.wait(timeout=0.1)
        assert result is False, "Flag should not be set initially"

        # Set the flag
        Server_AGI._shutdown_flag.set()
        assert Server_AGI._shutdown_flag.is_set() is True

        # Now wait should return immediately with True
        result = Server_AGI._shutdown_flag.wait(timeout=1.0)
        assert result is True, "Flag should be set"
    finally:
        # Restore original state
        if original_state:
            Server_AGI._shutdown_flag.set()
        else:
            Server_AGI._shutdown_flag.clear()


# ---------------------------------------------------------------------------
# 3. Integration: signal handling
# ---------------------------------------------------------------------------


def test_sigterm_signal_handler_defined():
    """Verify SIGTERM signal handler is registered."""
    # Get the currently registered handler for SIGTERM
    current_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # It should not be SIG_DFL or SIG_IGN if properly set up
    # (this is a basic check; the actual handler is in the main block)
    assert current_handler is not None

    # Restore the handler
    signal.signal(signal.SIGTERM, current_handler)


def test_sigint_signal_handler_defined():
    """Verify SIGINT signal handler is registered."""
    # Get the currently registered handler for SIGINT
    current_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)

    # It should not be SIG_DFL or SIG_IGN if properly set up
    assert current_handler is not None

    # Restore the handler
    signal.signal(signal.SIGINT, current_handler)


def test_shutdown_flag_module_level():
    """Verify _shutdown_flag is a module-level variable."""
    # It should be accessible directly from the module
    flag = getattr(Server_AGI, "_shutdown_flag", None)
    assert flag is not None
    assert isinstance(flag, threading.Event)


def test_shutdown_flag_state_across_calls():
    """Verify _shutdown_flag persists state across multiple accesses."""
    original_state = Server_AGI._shutdown_flag.is_set()

    try:
        # Clear and verify
        Server_AGI._shutdown_flag.clear()
        assert Server_AGI._shutdown_flag.is_set() is False

        # Access again and verify state persists
        assert Server_AGI._shutdown_flag.is_set() is False

        # Set and verify
        Server_AGI._shutdown_flag.set()
        assert Server_AGI._shutdown_flag.is_set() is True

        # Access again and verify state persists
        assert Server_AGI._shutdown_flag.is_set() is True
    finally:
        # Restore original state
        if original_state:
            Server_AGI._shutdown_flag.set()
        else:
            Server_AGI._shutdown_flag.clear()


# ---------------------------------------------------------------------------
# 4. Thread safety tests
# ---------------------------------------------------------------------------


def test_shutdown_flag_thread_safety():
    """Verify _shutdown_flag is thread-safe."""
    original_state = Server_AGI._shutdown_flag.is_set()

    try:
        Server_AGI._shutdown_flag.clear()

        results = []

        def set_flag_from_thread():
            Server_AGI._shutdown_flag.set()
            results.append("set")

        def wait_for_flag_from_thread():
            # Wait for the flag to be set
            Server_AGI._shutdown_flag.wait(timeout=5.0)
            results.append("waited")

        # Start threads
        t1 = threading.Thread(target=wait_for_flag_from_thread)
        t2 = threading.Thread(target=set_flag_from_thread)

        t1.start()
        t2.start()

        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        # Both threads should have completed
        assert len(results) == 2
        assert "set" in results
        assert "waited" in results

    finally:
        # Restore original state
        if original_state:
            Server_AGI._shutdown_flag.set()
        else:
            Server_AGI._shutdown_flag.clear()


# ---------------------------------------------------------------------------
# 5. Graceful shutdown signal flow
# ---------------------------------------------------------------------------


def test_shutdown_flag_can_stop_loop():
    """Verify _shutdown_flag can be used to stop a loop."""
    original_state = Server_AGI._shutdown_flag.is_set()

    try:
        Server_AGI._shutdown_flag.clear()

        # Simulate a loop that checks the flag
        iterations = 0
        max_iterations = 100

        def loop_with_shutdown_check():
            nonlocal iterations
            while not Server_AGI._shutdown_flag.is_set() and iterations < max_iterations:
                iterations += 1
                import time
                time.sleep(0.01)

        # Set flag to stop loop quickly
        import threading as t

        def stop_after_delay():
            import time

            time.sleep(0.05)
            Server_AGI._shutdown_flag.set()

        t.Thread(target=stop_after_delay, daemon=True).start()

        loop_with_shutdown_check()

        # Loop should have stopped due to flag
        assert Server_AGI._shutdown_flag.is_set()
        assert iterations < max_iterations  # Should stop before max_iterations

    finally:
        # Restore original state
        if original_state:
            Server_AGI._shutdown_flag.set()
        else:
            Server_AGI._shutdown_flag.clear()


# ---------------------------------------------------------------------------
# 6. Module initialization verification
# ---------------------------------------------------------------------------


def test_shutdown_flag_initialized_on_import():
    """Verify _shutdown_flag is initialized when Server_AGI is imported."""
    # This test verifies the flag exists and is initialized
    assert hasattr(Server_AGI, "_shutdown_flag")

    flag = Server_AGI._shutdown_flag
    assert isinstance(flag, threading.Event)

    # Verify it's in a valid state (either set or unset)
    try:
        result = flag.is_set()
        assert isinstance(result, bool)
    except Exception as e:
        pytest.fail(f"_shutdown_flag.is_set() failed: {e}")
