from __future__ import annotations

import uuid

import pytest
from gw_support import register_and_login


def _create(client, headers, features, machine_id="press-07", **extra_headers):
    return client.post(
        "/api/v1/predictions",
        json={"machine_id": machine_id, "features": features},
        headers={**headers, **extra_headers},
    )


def test_prediction_end_to_end(gateway, sample_features):
    headers = register_and_login(gateway, "op@example.com")
    resp = _create(gateway, headers, sample_features)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert body["model_version"].startswith("rf-")
    assert gateway.get(f"/api/v1/predictions/{body['id']}", headers=headers).json() == body


def test_new_inputs_are_stored_without_schema_change(gateway, sample_features):
    headers = register_and_login(gateway, "humid@example.com")
    body = _create(gateway, headers, {**sample_features, "humidity_pct": 63.5}).json()
    assert body["features"]["humidity_pct"] == 63.5


def test_request_id_is_propagated_and_stored(gateway, sample_features):
    headers = register_and_login(gateway, "trace@example.com")
    resp = _create(gateway, headers, sample_features, **{"X-Request-ID": "req-0123456789"})
    assert resp.headers["x-request-id"] == "req-0123456789"
    assert resp.json()["request_id"] == "req-0123456789"


def test_users_only_see_their_own_predictions(gateway, sample_features):
    alice = register_and_login(gateway, "alice@example.com")
    bob = register_and_login(gateway, "bob@example.com")
    alice_pred = _create(gateway, alice, sample_features).json()
    _create(gateway, bob, sample_features)

    assert gateway.get("/api/v1/predictions", headers=alice).json()["total"] == 1
    url = f"/api/v1/predictions/{alice_pred['id']}"
    assert gateway.get(url, headers=bob).status_code == 404
    assert gateway.delete(url, headers=bob).status_code == 404
    assert gateway.get(url, headers=alice).status_code == 200


def test_delete_own_prediction(gateway, sample_features):
    headers = register_and_login(gateway, "del@example.com")
    pred = _create(gateway, headers, sample_features).json()
    assert gateway.delete(f"/api/v1/predictions/{pred['id']}", headers=headers).status_code == 204
    assert gateway.get(f"/api/v1/predictions/{pred['id']}", headers=headers).status_code == 404


def test_listing_filters_and_paginates(gateway, sample_features):
    headers = register_and_login(gateway, "pages@example.com")
    for i in range(3):
        _create(gateway, headers, sample_features, machine_id=f"pump-{i}")
    page = gateway.get("/api/v1/predictions?limit=2&offset=0", headers=headers).json()
    assert page["total"] == 3
    assert len(page["items"]) == 2
    only = gateway.get("/api/v1/predictions?machine_id=pump-1", headers=headers).json()
    assert [p["machine_id"] for p in only["items"]] == ["pump-1"]
    assert gateway.get("/api/v1/predictions?limit=1000", headers=headers).status_code == 422


def test_model_contract_errors_are_returned_as_422(gateway, sample_features):
    headers = register_and_login(gateway, "bad@example.com")
    resp = _create(gateway, headers, {**sample_features, "load_pct": 999})
    assert resp.status_code == 422
    assert "load_pct" in str(resp.json()["detail"])


@pytest.mark.parametrize(
    "payload",
    [
        {"machine_id": "<script>alert(1)</script>", "features": {"rpm": 1}},
        {"machine_id": "ok-1", "features": {"machine_type": "<img src=x onerror=alert(1)>"}},
        {"machine_id": "ok-1", "features": {"Bad Key": 1}},
        {"machine_id": "ok-1", "features": {}},
        {"machine_id": "ok-1", "features": {"rpm": True}},
    ],
    ids=["xss-machine-id", "xss-feature", "bad-key", "empty", "bool"],
)
def test_malicious_or_malformed_input_rejected(gateway, payload):
    headers = register_and_login(gateway, "fuzz@example.com")
    assert gateway.post("/api/v1/predictions", json=payload, headers=headers).status_code == 422


def test_invalid_uuid_and_missing_auth(gateway):
    headers = register_and_login(gateway, "uuid@example.com")
    assert gateway.get("/api/v1/predictions/not-a-uuid", headers=headers).status_code == 422
    assert gateway.get(f"/api/v1/predictions/{uuid.uuid4()}").status_code == 401
