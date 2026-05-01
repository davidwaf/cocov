"""
webapp/main.py
--------------
Reviewer web application for the COCOV framework.

Provides a browser-based interface through which a human
reviewer can inspect escalated verification observations
and provide identity decisions. Reviewer responses are
logged and fed back into the COCOV prototype update
pipeline.

Escalation occurs when an incoming probe embedding falls
outside automatic acceptance bounds --- either similarity
is below tau_ver or drift exceeds tau_delta. In these
cases, the prototype update is suspended pending a
reviewer decision.

The reviewer is presented with:
    - The probe face image
    - The claimed identity and its enrolled reference image
    - The computed similarity and drift scores
    - Four response options:
        1. Confirm claimed identity
        2. Assign different enrolled identity
        3. Create new identity
        4. Reject observation

All reviewer decisions are logged with timestamps,
response times, and decision metadata for audit and
analysis.

This application is implemented using FastAPI and serves
a minimal HTML/JavaScript frontend. It is designed for
single-reviewer operation in a research setting.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import json
import time
import logging
import base64
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Application Setup
# ----------------------------------------------------------

app = FastAPI(
    title="COCOV Reviewer Interface",
    description=(
        "Human reviewer interface for Continuous "
        "Collaborative Verification escalated observations."
    ),
    version="1.0.0"
)

# In-memory queue of escalated observations
# In production this would be a persistent queue
escalation_queue: list[dict] = []
reviewer_log: list[dict] = []

# Log file path
LOG_PATH = Path("/opt/data/logs/reviewer_log.json")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# Data Models
# ----------------------------------------------------------

class EscalatedObservation(BaseModel):
    """
    An observation escalated for reviewer inspection.

    Attributes
    ----------
    observation_id : str
        Unique identifier for this escalation event.
    claimed_identity_id : str
        Identity claimed by the probe.
    similarity : float
        Cosine similarity score at time of escalation.
    drift : float
        Cosine distance from identity centre.
    escalation_reason : str
        Either 'low_similarity', 'high_drift', or 'both'.
    probe_image_path : str
        Path to the probe face image.
    enrolled_image_path : str
        Path to a reference image for the claimed identity.
    sequence_position : int
        Position of this event in the evaluation stream.
    timestamp : str
        ISO format timestamp of escalation.
    """
    observation_id: str
    claimed_identity_id: str
    similarity: float
    drift: float
    escalation_reason: str
    probe_image_path: str
    enrolled_image_path: str
    sequence_position: int
    timestamp: str


class ReviewerDecision(BaseModel):
    """
    A reviewer decision for an escalated observation.

    Attributes
    ----------
    observation_id : str
        ID of the observation being decided.
    action : str
        One of: 'confirm', 'assign', 'create', 'reject'.
    assigned_identity_id : str, optional
        If action is 'assign', the correct identity ID.
    new_identity_name : str, optional
        If action is 'create', name for the new identity.
    reviewer_notes : str, optional
        Optional reviewer comments.
    response_time_ms : int
        Time taken by reviewer to decide in milliseconds.
    """
    observation_id: str
    action: str
    assigned_identity_id: Optional[str] = None
    new_identity_name: Optional[str] = None
    reviewer_notes: Optional[str] = None
    response_time_ms: int = 0


# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------

def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string for browser display.

    Parameters
    ----------
    image_path : str
        Path to image file.

    Returns
    -------
    str
        Base64-encoded image string with data URI prefix.
        Returns empty string if file not found.
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning(f"Image not found: {image_path}")
        return ""

    with open(path, 'rb') as f:
        data = f.read()

    ext = path.suffix.lower().lstrip('.')
    if ext == 'jpg':
        ext = 'jpeg'

    encoded = base64.b64encode(data).decode('utf-8')
    return f"data:image/{ext};base64,{encoded}"


def determine_escalation_reason(
    similarity: float,
    drift: float,
    tau_ver: float,
    tau_delta: float
) -> str:
    """
    Determine the reason for escalation.

    Parameters
    ----------
    similarity : float
        Cosine similarity score.
    drift : float
        Cosine distance from identity centre.
    tau_ver : float
        Verification threshold.
    tau_delta : float
        Drift threshold.

    Returns
    -------
    str
        Escalation reason string.
    """
    low_sim = similarity < tau_ver
    high_drift = drift > tau_delta

    if low_sim and high_drift:
        return 'both'
    elif low_sim:
        return 'low_similarity'
    elif high_drift:
        return 'high_drift'
    else:
        return 'none'


def log_decision(decision: ReviewerDecision) -> None:
    """
    Append reviewer decision to the audit log.

    Parameters
    ----------
    decision : ReviewerDecision
        Completed reviewer decision to log.
    """
    entry = {
        'observation_id': decision.observation_id,
        'action': decision.action,
        'assigned_identity_id': decision.assigned_identity_id,
        'new_identity_name': decision.new_identity_name,
        'reviewer_notes': decision.reviewer_notes,
        'response_time_ms': decision.response_time_ms,
        'logged_at': datetime.now().isoformat()
    }

    reviewer_log.append(entry)

    # Persist to disk
    with open(LOG_PATH, 'w') as f:
        json.dump(reviewer_log, f, indent=2)

    logger.info(
        f"Decision logged: {decision.observation_id} "
        f"-> {decision.action}"
    )


# ----------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def reviewer_interface() -> HTMLResponse:
    """
    Serve the reviewer interface HTML page.

    Returns the main reviewer dashboard that displays
    escalated observations and collects decisions.
    """
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" 
          content="width=device-width, initial-scale=1.0">
    <title>COCOV Reviewer Interface</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont,
                         'Segoe UI', sans-serif;
            background: #f5f5f5;
            color: #333;
        }

        header {
            background: #7B1FA2;
            color: white;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        header h1 {
            font-size: 1.2rem;
            font-weight: 600;
        }

        #queue-count {
            background: white;
            color: #7B1FA2;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        main {
            max-width: 960px;
            margin: 32px auto;
            padding: 0 16px;
        }

        #empty-state {
            text-align: center;
            padding: 64px 32px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        #empty-state h2 {
            font-size: 1.3rem;
            color: #7B1FA2;
            margin-bottom: 8px;
        }

        #empty-state p {
            color: #888;
            font-size: 0.95rem;
        }

        #observation-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
            display: none;
        }

        .card-header {
            background: #f8f4fc;
            padding: 16px 24px;
            border-bottom: 1px solid #e8e0f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-header h2 {
            font-size: 1rem;
            color: #7B1FA2;
        }

        .escalation-badge {
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .badge-both {
            background: #ffebee;
            color: #c62828;
        }

        .badge-low_similarity {
            background: #fff3e0;
            color: #e65100;
        }

        .badge-high_drift {
            background: #fce4ec;
            color: #880e4f;
        }

        .images-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            padding: 24px;
        }

        .image-panel {
            text-align: center;
        }

        .image-panel h3 {
            font-size: 0.85rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .image-panel img {
            width: 100%;
            max-width: 320px;
            height: 320px;
            object-fit: cover;
            border-radius: 8px;
            border: 2px solid #e8e0f0;
        }

        .scores-section {
            padding: 0 24px 24px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .score-card {
            background: #f8f4fc;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }

        .score-card .label {
            font-size: 0.78rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }

        .score-card .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #7B1FA2;
        }

        .score-card .sub {
            font-size: 0.75rem;
            color: #aaa;
            margin-top: 2px;
        }

        .identity-info {
            padding: 0 24px 24px;
            background: #fafafa;
            border-top: 1px solid #f0f0f0;
        }

        .identity-info p {
            font-size: 0.9rem;
            color: #555;
            padding: 12px 0;
        }

        .identity-info strong {
            color: #333;
        }

        .decision-section {
            padding: 24px;
            border-top: 1px solid #f0f0f0;
        }

        .decision-section h3 {
            font-size: 0.9rem;
            color: #555;
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .decision-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .decision-btn {
            padding: 14px 16px;
            border: 2px solid transparent;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
            text-align: left;
        }

        .decision-btn .btn-title {
            display: block;
            font-size: 0.95rem;
        }

        .decision-btn .btn-sub {
            display: block;
            font-size: 0.75rem;
            font-weight: 400;
            opacity: 0.7;
            margin-top: 2px;
        }

        .btn-confirm {
            background: #e8f5e9;
            color: #2e7d32;
            border-color: #a5d6a7;
        }

        .btn-confirm:hover {
            background: #c8e6c9;
        }

        .btn-assign {
            background: #e3f2fd;
            color: #1565c0;
            border-color: #90caf9;
        }

        .btn-assign:hover {
            background: #bbdefb;
        }

        .btn-create {
            background: #fff3e0;
            color: #e65100;
            border-color: #ffcc80;
        }

        .btn-create:hover {
            background: #ffe0b2;
        }

        .btn-reject {
            background: #ffebee;
            color: #c62828;
            border-color: #ef9a9a;
        }

        .btn-reject:hover {
            background: #ffcdd2;
        }

        .notes-field {
            width: 100%;
            margin-top: 16px;
            padding: 10px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 0.9rem;
            font-family: inherit;
            resize: vertical;
            min-height: 60px;
        }

        .timer {
            font-size: 0.8rem;
            color: #aaa;
            margin-top: 12px;
            text-align: right;
        }

        #status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #333;
            color: white;
            padding: 10px 24px;
            font-size: 0.85rem;
            display: none;
        }
    </style>
</head>
<body>

<header>
    <h1>COCOV Reviewer Interface</h1>
    <span id="queue-count">Loading...</span>
</header>

<main>
    <div id="empty-state">
        <h2>No pending observations</h2>
        <p>Escalated observations will appear here 
           for review.</p>
    </div>

    <div id="observation-card">
        <div class="card-header">
            <h2 id="obs-id">Observation</h2>
            <span id="escalation-badge" 
                  class="escalation-badge"></span>
        </div>

        <div class="images-section">
            <div class="image-panel">
                <h3>Probe Image</h3>
                <img id="probe-img" src="" alt="Probe">
            </div>
            <div class="image-panel">
                <h3>Enrolled Reference</h3>
                <img id="enrolled-img" src="" 
                     alt="Enrolled">
            </div>
        </div>

        <div class="scores-section">
            <div class="score-card">
                <div class="label">Similarity</div>
                <div class="value" id="sim-value">-</div>
                <div class="sub">cosine similarity</div>
            </div>
            <div class="score-card">
                <div class="label">Drift</div>
                <div class="value" id="drift-value">-</div>
                <div class="sub">cosine distance</div>
            </div>
        </div>

        <div class="identity-info">
            <p>
                <strong>Claimed identity:</strong>
                <span id="claimed-id">-</span>
            </p>
            <p>
                <strong>Stream position:</strong>
                <span id="seq-pos">-</span>
            </p>
            <p>
                <strong>Escalated at:</strong>
                <span id="esc-time">-</span>
            </p>
        </div>

        <div class="decision-section">
            <h3>Your Decision</h3>
            <div class="decision-buttons">
                <button class="decision-btn btn-confirm"
                        onclick="decide('confirm')">
                    <span class="btn-title">
                        Confirm Identity
                    </span>
                    <span class="btn-sub">
                        Probe matches claimed identity
                    </span>
                </button>
                <button class="decision-btn btn-assign"
                        onclick="decide('assign')">
                    <span class="btn-title">
                        Assign Different Identity
                    </span>
                    <span class="btn-sub">
                        Probe belongs to another 
                        enrolled identity
                    </span>
                </button>
                <button class="decision-btn btn-create"
                        onclick="decide('create')">
                    <span class="btn-title">
                        Create New Identity
                    </span>
                    <span class="btn-sub">
                        Probe is a new unenrolled person
                    </span>
                </button>
                <button class="decision-btn btn-reject"
                        onclick="decide('reject')">
                    <span class="btn-title">
                        Reject Observation
                    </span>
                    <span class="btn-sub">
                        Unusable image or unclear identity
                    </span>
                </button>
            </div>

            <textarea class="notes-field"
                      id="reviewer-notes"
                      placeholder="Optional notes...">
            </textarea>

            <div class="timer" id="timer">
                Time: 0s
            </div>
        </div>
    </div>
</main>

<div id="status-bar"></div>

<script>
    let currentObs = null;
    let startTime = null;
    let timerInterval = null;
    let currentObsId = null;

    // Fetch next observation from queue
    async function fetchNext() {
        try {
            const resp = await fetch('/queue/next');
            const data = await resp.json();

            updateQueueCount();

            if (!data.observation_id) {
                showEmpty();
                return;
            }

            currentObs = data;
            currentObsId = data.observation_id;
            displayObservation(data);
        } catch(e) {
            console.error('Failed to fetch:', e);
        }
    }

    function displayObservation(obs) {
        document.getElementById('empty-state')
            .style.display = 'none';
        document.getElementById('observation-card')
            .style.display = 'block';

        document.getElementById('obs-id').textContent =
            'Observation ' + obs.sequence_position;
        document.getElementById('claimed-id').textContent =
            obs.claimed_identity_id;
        document.getElementById('seq-pos').textContent =
            obs.sequence_position;
        document.getElementById('esc-time').textContent =
            obs.timestamp;
        document.getElementById('sim-value').textContent =
            obs.similarity.toFixed(4);
        document.getElementById('drift-value').textContent =
            obs.drift.toFixed(4);

        // Escalation badge
        const badge = document.getElementById(
            'escalation-badge'
        );
        badge.className =
            'escalation-badge badge-' + obs.escalation_reason;
        const reasons = {
            'both': 'Low Similarity + High Drift',
            'low_similarity': 'Low Similarity',
            'high_drift': 'High Drift'
        };
        badge.textContent =
            reasons[obs.escalation_reason] || 'Escalated';

        // Images
        if (obs.probe_image_b64) {
            document.getElementById('probe-img').src =
                obs.probe_image_b64;
        }
        if (obs.enrolled_image_b64) {
            document.getElementById('enrolled-img').src =
                obs.enrolled_image_b64;
        }

        // Start timer
        startTime = Date.now();
        if (timerInterval) clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            const elapsed = Math.round(
                (Date.now() - startTime) / 1000
            );
            document.getElementById('timer').textContent =
                'Time: ' + elapsed + 's';
        }, 1000);

        document.getElementById('reviewer-notes').value = '';
    }

    async function decide(action) {
        if (!currentObsId) return;

        const responseTimeMs = startTime
            ? Date.now() - startTime
            : 0;

        let assignedId = null;
        let newName = null;

        if (action === 'assign') {
            assignedId = prompt(
                'Enter the correct enrolled identity ID:'
            );
            if (!assignedId) return;
        }

        if (action === 'create') {
            newName = prompt(
                'Enter a name for the new identity:'
            );
            if (!newName) return;
        }

        const notes = document.getElementById(
            'reviewer-notes'
        ).value;

        try {
            const resp = await fetch('/decision', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    observation_id: currentObsId,
                    action: action,
                    assigned_identity_id: assignedId,
                    new_identity_name: newName,
                    reviewer_notes: notes,
                    response_time_ms: responseTimeMs
                })
            });

            if (resp.ok) {
                showStatus(
                    'Decision recorded: ' + action,
                    '#4CAF50'
                );
                if (timerInterval) {
                    clearInterval(timerInterval);
                }
                setTimeout(fetchNext, 800);
            } else {
                showStatus('Error recording decision', '#F44336');
            }
        } catch(e) {
            showStatus('Connection error', '#F44336');
        }
    }

    async function updateQueueCount() {
        try {
            const resp = await fetch('/queue/count');
            const data = await resp.json();
            document.getElementById('queue-count').textContent =
                data.count + ' pending';
        } catch(e) {
            document.getElementById('queue-count').textContent =
                'Queue unknown';
        }
    }

    function showEmpty() {
        document.getElementById('empty-state')
            .style.display = 'block';
        document.getElementById('observation-card')
            .style.display = 'none';
        currentObs = null;
        currentObsId = null;
    }

    function showStatus(message, color) {
        const bar = document.getElementById('status-bar');
        bar.textContent = message;
        bar.style.background = color;
        bar.style.display = 'block';
        setTimeout(() => {
            bar.style.display = 'none';
        }, 2000);
    }

    // Poll for new observations every 3 seconds
    fetchNext();
    setInterval(fetchNext, 3000);
    setInterval(updateQueueCount, 5000);
</script>
</body>
</html>
    """
    return HTMLResponse(content=html)


@app.get("/queue/next")
async def get_next_observation() -> JSONResponse:
    """
    Return the next escalated observation from the queue.

    Returns
    -------
    JSONResponse
        Next observation with base64-encoded images,
        or empty dict if queue is empty.
    """
    if not escalation_queue:
        return JSONResponse({})

    obs = escalation_queue[0]

    # Encode images for browser display
    obs_response = dict(obs)
    obs_response['probe_image_b64'] = image_to_base64(
        obs.get('probe_image_path', '')
    )
    obs_response['enrolled_image_b64'] = image_to_base64(
        obs.get('enrolled_image_path', '')
    )

    return JSONResponse(obs_response)


@app.get("/queue/count")
async def get_queue_count() -> JSONResponse:
    """
    Return the number of pending escalated observations.

    Returns
    -------
    JSONResponse
        Queue count.
    """
    return JSONResponse({'count': len(escalation_queue)})


@app.post("/queue/add")
async def add_to_queue(
    observation: EscalatedObservation
) -> JSONResponse:
    """
    Add an escalated observation to the reviewer queue.

    Called by the COCOV system when an observation
    triggers the escalation condition.

    Parameters
    ----------
    observation : EscalatedObservation
        Observation data for reviewer inspection.

    Returns
    -------
    JSONResponse
        Confirmation with observation ID.
    """
    obs_dict = observation.model_dump()
    escalation_queue.append(obs_dict)

    logger.info(
        f"Observation {observation.observation_id} "
        f"added to queue. Queue size: "
        f"{len(escalation_queue)}"
    )

    return JSONResponse({
        'status': 'queued',
        'observation_id': observation.observation_id,
        'queue_position': len(escalation_queue)
    })


@app.post("/decision")
async def submit_decision(
    decision: ReviewerDecision
) -> JSONResponse:
    """
    Submit a reviewer decision for an escalated observation.

    Removes the observation from the queue and logs
    the decision. The COCOV system polls this endpoint
    or receives a webhook to retrieve decisions.

    Parameters
    ----------
    decision : ReviewerDecision
        Reviewer decision data.

    Returns
    -------
    JSONResponse
        Confirmation with decision summary.

    Raises
    ------
    HTTPException
        If the observation ID is not found in the queue.
    """
    # Validate action
    valid_actions = {'confirm', 'assign', 'create', 'reject'}
    if decision.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {decision.action}. "
                   f"Must be one of {valid_actions}."
        )

    # Find and remove from queue
    obs_found = False
    for i, obs in enumerate(escalation_queue):
        if obs['observation_id'] == decision.observation_id:
            escalation_queue.pop(i)
            obs_found = True
            break

    if not obs_found:
        raise HTTPException(
            status_code=404,
            detail=f"Observation {decision.observation_id} "
                   f"not found in queue."
        )

    # Log decision
    log_decision(decision)

    return JSONResponse({
        'status': 'recorded',
        'observation_id': decision.observation_id,
        'action': decision.action,
        'queue_remaining': len(escalation_queue)
    })


@app.get("/decisions/{observation_id}")
async def get_decision(observation_id: str) -> JSONResponse:
    """
    Retrieve a logged decision by observation ID.

    Used by the COCOV system to poll for reviewer
    responses to specific escalated observations.

    Parameters
    ----------
    observation_id : str
        ID of the observation to retrieve decision for.

    Returns
    -------
    JSONResponse
        Decision data if found, or pending status.
    """
    for entry in reviewer_log:
        if entry['observation_id'] == observation_id:
            return JSONResponse({
                'status': 'decided',
                'decision': entry
            })

    # Check if still in queue
    for obs in escalation_queue:
        if obs['observation_id'] == observation_id:
            return JSONResponse({
                'status': 'pending',
                'queue_position': escalation_queue.index(obs)
            })

    return JSONResponse({
        'status': 'not_found',
        'observation_id': observation_id
    })


@app.get("/log")
async def get_reviewer_log() -> JSONResponse:
    """
    Return the complete reviewer decision log.

    Returns
    -------
    JSONResponse
        All logged reviewer decisions.
    """
    return JSONResponse({
        'n_decisions': len(reviewer_log),
        'decisions': reviewer_log
    })


@app.get("/stats")
async def get_reviewer_stats() -> JSONResponse:
    """
    Return reviewer performance statistics.

    Returns
    -------
    JSONResponse
        Decision counts, mean response time,
        and action distribution.
    """
    if not reviewer_log:
        return JSONResponse({
            'n_decisions': 0,
            'mean_response_time_ms': 0,
            'action_counts': {}
        })

    response_times = [
        e['response_time_ms'] for e in reviewer_log
    ]
    action_counts = {}
    for entry in reviewer_log:
        action = entry['action']
        action_counts[action] = (
            action_counts.get(action, 0) + 1
        )

    return JSONResponse({
        'n_decisions': len(reviewer_log),
        'mean_response_time_ms': float(
            np.mean(response_times)
        ),
        'median_response_time_ms': float(
            np.median(response_times)
        ),
        'action_counts': action_counts,
        'queue_size': len(escalation_queue)
    })


@app.get("/health")
async def health_check() -> JSONResponse:
    """
    Health check endpoint.

    Returns
    -------
    JSONResponse
        Service status.
    """
    return JSONResponse({
        'status': 'healthy',
        'queue_size': len(escalation_queue),
        'n_decisions': len(reviewer_log)
    })


# ----------------------------------------------------------
# Helper for COCOV Integration
# ----------------------------------------------------------

def create_escalation(
    claimed_identity_id: str,
    similarity: float,
    drift: float,
    probe_image_path: str,
    enrolled_image_path: str,
    sequence_position: int,
    tau_ver: float,
    tau_delta: float
) -> dict:
    """
    Create a structured escalation event for the queue.

    Called by the COCOV system when an observation
    meets the escalation condition. Returns a dict
    suitable for posting to /queue/add.

    Parameters
    ----------
    claimed_identity_id : str
        Identity claimed by the probe.
    similarity : float
        Cosine similarity score.
    drift : float
        Cosine distance from identity centre.
    probe_image_path : str
        Path to probe image.
    enrolled_image_path : str
        Path to enrolled reference image.
    sequence_position : int
        Stream position of this event.
    tau_ver : float
        Verification threshold for reason determination.
    tau_delta : float
        Drift threshold for reason determination.

    Returns
    -------
    dict
        Escalation event dictionary.
    """
    return {
        'observation_id': str(uuid.uuid4()),
        'claimed_identity_id': claimed_identity_id,
        'similarity': similarity,
        'drift': drift,
        'escalation_reason': determine_escalation_reason(
            similarity, drift, tau_ver, tau_delta
        ),
        'probe_image_path': probe_image_path,
        'enrolled_image_path': enrolled_image_path,
        'sequence_position': sequence_position,
        'timestamp': datetime.now().isoformat()
    }


# ----------------------------------------------------------
# Entry Point
# ----------------------------------------------------------

if __name__ == '__main__':
    import uvicorn
    import yaml

    # Load config for host/port
    config_path = Path(
        '/opt/code/cocov/config/config.yaml'
    )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    webapp_cfg = config.get('webapp', {})
    host = webapp_cfg.get('host', '0.0.0.0')
    port = webapp_cfg.get('port', 8000)

    logging.basicConfig(level=logging.INFO)
    logger.info(
        f"Starting COCOV Reviewer Interface at "
        f"http://{host}:{port}"
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level='info'
    )
