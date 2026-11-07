from fastapi.testclient import TestClient

from androidmirror.server import create_app


def test_password_guards_every_route(tmp_path):
    c = TestClient(create_app(tmp_path, password="s3cret"))
    r = c.get("/api/status")
    assert r.status_code == 401 and "Basic" in r.headers["www-authenticate"]
    assert c.get("/", auth=("x", "wrong")).status_code == 401
    assert c.get("/api/status", auth=("anyone", "s3cret")).status_code == 200


def test_no_password_on_localhost(tmp_path):
    assert TestClient(create_app(tmp_path)).get("/api/status").status_code == 200
