"""
Finite State Machine (FSM) implementation inspired by NautilusTrader.

Provides a thread-safe FSM for managing trading bot state transitions with:
- Explicit state transition table
- Invalid transition error handling (fail-fast)
- State callbacks (on_enter, on_exit)
- Thread safety for live bot usage

Trading states: SCANNING → STALKING → ENTRY_READY → IN_TRADE → EXITING → COOLDOWN
"""

import logging
import threading
from typing import Dict, Tuple, Optional, Callable, Any
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


# ─── State Definitions ───────────────────────────────────────────────

class TradingState(Enum):
    """Trading bot states in order of normal progression."""
    SCANNING = auto()      # Looking for regime/signals
    STALKING = auto()      # Watching specific setup develop
    ENTRY_READY = auto()   # Signal confirmed, ready to enter
    IN_TRADE = auto()      # Position active
    EXITING = auto()       # Closing position
    COOLDOWN = auto()      # Temporary rest after trade


class Trigger(Enum):
    """Events that can trigger state transitions."""
    # Signal triggers
    SIGNAL_DETECTED = auto()        # Scanning → Stalking
    SIGNAL_CONFIRMED = auto()       # Stalking → Entry_Ready  
    ENTRY_EXECUTED = auto()         # Entry_Ready → In_Trade
    
    # Exit triggers
    EXIT_SIGNAL = auto()            # In_Trade → Exiting
    STOP_HIT = auto()              # In_Trade → Exiting
    TARGET_HIT = auto()            # In_Trade → Exiting
    EXIT_EXECUTED = auto()         # Exiting → Cooldown
    
    # Reset triggers
    SIGNAL_LOST = auto()           # Stalking/Entry_Ready → Scanning
    COOLDOWN_EXPIRED = auto()      # Cooldown → Scanning
    
    # Error triggers
    ERROR_RECOVERY = auto()        # Any state → Scanning
    EMERGENCY_EXIT = auto()        # Any state → Exiting


# ─── FSM Exception ───────────────────────────────────────────────────

class InvalidStateTransition(Exception):
    """Raised when an invalid state transition is attempted."""
    
    def __init__(self, current_state: TradingState, trigger: Trigger):
        self.current_state = current_state
        self.trigger = trigger
        super().__init__(
            f"Invalid transition: {current_state.name} -> {trigger.name}"
        )


# ─── State Transition Context ────────────────────────────────────────

@dataclass
class StateTransitionContext:
    """Context data passed to state callbacks."""
    from_state: TradingState
    to_state: TradingState
    trigger: Trigger
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None


# ─── Finite State Machine ────────────────────────────────────────────

class FiniteStateMachine:
    """Thread-safe finite state machine for trading bot state management.
    
    Based on NautilusTrader's FSM pattern with explicit transition table
    and state callbacks. Invalid transitions raise errors (fail-fast).
    """
    
    def __init__(self, initial_state: TradingState = TradingState.SCANNING):
        self._lock = threading.RLock()
        self._state = initial_state
        self._transition_count = 0
        self._transition_history: list[StateTransitionContext] = []
        
        # State callbacks
        self._on_enter_callbacks: Dict[TradingState, list[Callable]] = {}
        self._on_exit_callbacks: Dict[TradingState, list[Callable]] = {}
        
        # Initialize callback dictionaries
        for state in TradingState:
            self._on_enter_callbacks[state] = []
            self._on_exit_callbacks[state] = []
        
        # Build transition table
        self._transition_table = self._build_transition_table()
        
        logger.info(f"FSM initialized in state: {self._state.name}")
    
    def _build_transition_table(self) -> Dict[Tuple[TradingState, Trigger], TradingState]:
        """Build explicit state transition table.
        
        Returns dict with (current_state, trigger) -> next_state mappings.
        Invalid transitions are not included (will raise error).
        """
        return {
            # SCANNING transitions
            (TradingState.SCANNING, Trigger.SIGNAL_DETECTED): TradingState.STALKING,
            
            # STALKING transitions  
            (TradingState.STALKING, Trigger.SIGNAL_CONFIRMED): TradingState.ENTRY_READY,
            (TradingState.STALKING, Trigger.SIGNAL_LOST): TradingState.SCANNING,
            
            # ENTRY_READY transitions
            (TradingState.ENTRY_READY, Trigger.ENTRY_EXECUTED): TradingState.IN_TRADE,
            (TradingState.ENTRY_READY, Trigger.SIGNAL_LOST): TradingState.SCANNING,
            
            # IN_TRADE transitions
            (TradingState.IN_TRADE, Trigger.EXIT_SIGNAL): TradingState.EXITING,
            (TradingState.IN_TRADE, Trigger.STOP_HIT): TradingState.EXITING,
            (TradingState.IN_TRADE, Trigger.TARGET_HIT): TradingState.EXITING,
            
            # EXITING transitions
            (TradingState.EXITING, Trigger.EXIT_EXECUTED): TradingState.COOLDOWN,
            
            # COOLDOWN transitions
            (TradingState.COOLDOWN, Trigger.COOLDOWN_EXPIRED): TradingState.SCANNING,
            
            # Emergency transitions (from any state)
            (TradingState.SCANNING, Trigger.ERROR_RECOVERY): TradingState.SCANNING,
            (TradingState.STALKING, Trigger.ERROR_RECOVERY): TradingState.SCANNING,
            (TradingState.ENTRY_READY, Trigger.ERROR_RECOVERY): TradingState.SCANNING,
            (TradingState.IN_TRADE, Trigger.ERROR_RECOVERY): TradingState.SCANNING,
            (TradingState.EXITING, Trigger.ERROR_RECOVERY): TradingState.SCANNING,
            (TradingState.COOLDOWN, Trigger.ERROR_RECOVERY): TradingState.SCANNING,
            
            (TradingState.SCANNING, Trigger.EMERGENCY_EXIT): TradingState.EXITING,
            (TradingState.STALKING, Trigger.EMERGENCY_EXIT): TradingState.EXITING,
            (TradingState.ENTRY_READY, Trigger.EMERGENCY_EXIT): TradingState.EXITING,
            (TradingState.IN_TRADE, Trigger.EMERGENCY_EXIT): TradingState.EXITING,
            (TradingState.COOLDOWN, Trigger.EMERGENCY_EXIT): TradingState.EXITING,
            # EXITING -> EXITING is allowed (idempotent)
            (TradingState.EXITING, Trigger.EMERGENCY_EXIT): TradingState.EXITING,
        }
    
    @property
    def state(self) -> TradingState:
        """Get current state (thread-safe)."""
        with self._lock:
            return self._state
    
    @property
    def state_name(self) -> str:
        """Get current state name as string."""
        return self.state.name
    
    @property
    def transition_count(self) -> int:
        """Get total number of transitions executed."""
        with self._lock:
            return self._transition_count
    
    def trigger(self, trigger: Trigger, data: Optional[Dict[str, Any]] = None) -> TradingState:
        """Execute state transition with given trigger.
        
        Args:
            trigger: Trigger to apply
            data: Optional context data for callbacks
            
        Returns:
            New state after transition
            
        Raises:
            InvalidStateTransition: If transition is not allowed
        """
        with self._lock:
            current_state = self._state
            
            # Look up next state in transition table
            transition_key = (current_state, trigger)
            next_state = self._transition_table.get(transition_key)
            
            if next_state is None:
                raise InvalidStateTransition(current_state, trigger)
            
            # Execute transition
            self._execute_transition(current_state, next_state, trigger, data)
            
            return self._state
    
    def _execute_transition(
        self, 
        from_state: TradingState, 
        to_state: TradingState, 
        trigger: Trigger,
        data: Optional[Dict[str, Any]]
    ) -> None:
        """Execute the state transition with callbacks."""
        context = StateTransitionContext(
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            timestamp=datetime.now(),
            data=data or {}
        )
        
        # Call exit callbacks for current state
        for callback in self._on_exit_callbacks[from_state]:
            try:
                callback(context)
            except Exception as e:
                logger.error(f"Error in on_exit callback for {from_state.name}: {e}")
        
        # Update state
        old_state = self._state
        self._state = to_state
        self._transition_count += 1
        
        # Store transition history (keep last 100)
        self._transition_history.append(context)
        if len(self._transition_history) > 100:
            self._transition_history = self._transition_history[-100:]
        
        # Call enter callbacks for new state
        for callback in self._on_enter_callbacks[to_state]:
            try:
                callback(context)
            except Exception as e:
                logger.error(f"Error in on_enter callback for {to_state.name}: {e}")
        
        # Log transition
        if old_state != to_state:
            logger.info(
                f"FSM transition: {from_state.name} -> {to_state.name} "
                f"(trigger: {trigger.name})"
            )
    
    def register_on_enter(self, state: TradingState, callback: Callable) -> None:
        """Register callback to be called when entering a state.
        
        Callback signature: callback(context: StateTransitionContext) -> None
        """
        with self._lock:
            self._on_enter_callbacks[state].append(callback)
    
    def register_on_exit(self, state: TradingState, callback: Callable) -> None:
        """Register callback to be called when exiting a state.
        
        Callback signature: callback(context: StateTransitionContext) -> None  
        """
        with self._lock:
            self._on_exit_callbacks[state].append(callback)
    
    def can_trigger(self, trigger: Trigger) -> bool:
        """Check if a trigger is valid for current state without executing."""
        with self._lock:
            return (self._state, trigger) in self._transition_table
    
    def get_valid_triggers(self) -> list[Trigger]:
        """Get list of valid triggers for current state."""
        with self._lock:
            return [
                trigger for (state, trigger) in self._transition_table.keys()
                if state == self._state
            ]
    
    def get_transition_history(self, limit: int = 10) -> list[StateTransitionContext]:
        """Get recent transition history."""
        with self._lock:
            return self._transition_history[-limit:]
    
    def reset(self, new_state: TradingState = TradingState.SCANNING) -> None:
        """Reset FSM to specified state (typically used for error recovery)."""
        with self._lock:
            old_state = self._state
            self._state = new_state
            logger.warning(f"FSM reset: {old_state.name} -> {new_state.name}")
    
    def __str__(self) -> str:
        """String representation."""
        return f"FSM(state={self._state.name}, transitions={self._transition_count})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        with self._lock:
            valid_triggers = [t.name for t in self.get_valid_triggers()]
            return (
                f"FiniteStateMachine(state={self._state.name}, "
                f"transitions={self._transition_count}, "
                f"valid_triggers={valid_triggers})"
            )