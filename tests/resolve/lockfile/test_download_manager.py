# Copyright 2022 Pex project contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import json
import os.path
from contextlib import contextmanager
from io import BytesIO

import pytest

from pex import hashing
from pex.artifact_url import ArtifactURL, Fingerprint
from pex.atomic_directory import AtomicDirectory
from pex.cache.dirs import CacheDir
from pex.common import safe_rmtree
from pex.fs.lock import FileLockStyle
from pex.hashing import Sha1Fingerprint, Sha256Fingerprint
from pex.pep_503 import ProjectName
from pex.resolve.locked_resolve import Artifact, FileArtifact, LocalProjectArtifact
from pex.resolve.lockfile import download_manager as download_manager_module
from pex.resolve.lockfile.download_manager import DownloadedArtifact, DownloadManager
from pex.result import Error, catch
from pex.typing import TYPE_CHECKING
from pex.variables import ENV, Variables

if TYPE_CHECKING:
    from typing import Any, List, Union

    import attr  # vendor:skip

    from pex.hashing import HintedDigest
else:
    from pex.third_party import attr


class FakeDownloadManager(DownloadManager[Artifact]):
    def __init__(
        self,
        content,  # type: bytes
        pex_root=ENV,  # type: Union[str, Variables]
    ):
        # type: (...) -> None
        super(FakeDownloadManager, self).__init__(pex_root=pex_root)
        self._content = content
        self._calls = []  # type: List[str]

    @property
    def save_calls(self):
        # type: () -> List[str]
        return self._calls

    def save(
        self,
        artifact,  # type: Artifact
        project_name,  # type: ProjectName
        dest_dir,  # type: str
        digest,  # type: HintedDigest
    ):
        # type: (...) -> Union[str, Error]
        self.save_calls.append(dest_dir)
        digest.update(self._content)
        return artifact.filename if isinstance(artifact, FileArtifact) else "foo"


@pytest.fixture
def expected_content():
    # type: () -> bytes
    return b"expected content"


@pytest.fixture
def project_name():
    # type: () -> ProjectName
    return ProjectName("foo")


@pytest.fixture
def artifact(expected_content):
    # type: (bytes) -> FileArtifact
    return FileArtifact(
        url=ArtifactURL.parse("file:///foo-1.0.tar.gz"),
        fingerprint=Fingerprint.from_stream(BytesIO(expected_content), algorithm="sha1"),
        filename="foo-1.0.tar.gz",
        verified=False,
    )


@pytest.fixture
def pex_root(tmpdir):
    # type: (Any) -> str
    return os.path.join(str(tmpdir), "pex_root")


@pytest.fixture
def download_manager(
    expected_content,  # type: bytes
    pex_root,  # type: Any
):
    # type: (...) -> FakeDownloadManager
    return FakeDownloadManager(content=expected_content, pex_root=pex_root)


def test_storage_cache(
    artifact,  # type: FileArtifact
    project_name,  # type: ProjectName
    download_manager,  # type: FakeDownloadManager
):
    # type: (...) -> None

    downloaded_artifact1 = download_manager.store(artifact, project_name)
    downloaded_artifact2 = download_manager.store(artifact, project_name)
    assert downloaded_artifact1 == downloaded_artifact2
    assert 1 == len(download_manager.save_calls)


def test_download_cache_uses_bsd_locks(download_manager):
    # type: (FakeDownloadManager) -> None

    assert FileLockStyle.BSD is download_manager._file_lock_style


def test_storage_version_2_is_compatible(
    artifact,  # type: FileArtifact
    project_name,  # type: ProjectName
    download_manager,  # type: FakeDownloadManager
):
    # type: (...) -> None

    downloaded_artifact = download_manager.store(artifact, project_name)
    metadata_file = DownloadedArtifact.metadata_filename(os.path.dirname(downloaded_artifact.path))
    with open(metadata_file) as fp:
        metadata = json.load(fp)
    metadata.pop("editable")
    metadata["version"] = DownloadedArtifact._LEGACY_METADATA_VERSION
    with open(metadata_file, "w") as fp:
        json.dump(metadata, fp)

    assert downloaded_artifact == download_manager.store(artifact, project_name)
    assert 1 == len(
        download_manager.save_calls
    ), "Version 2 metadata should be read in place instead of forcing a cache rebuild."


def test_storage_version_2_is_rebuilt_for_an_editable_local_project(
    artifact,  # type: FileArtifact
    project_name,  # type: ProjectName
    download_manager,  # type: FakeDownloadManager
):
    # type: (...) -> None

    local_project = LocalProjectArtifact(
        url=ArtifactURL.parse("file:///foo"),
        fingerprint=artifact.fingerprint,
        verified=True,
        directory="/foo",
        editable=False,
    )
    downloaded_artifact = download_manager.store(local_project, project_name)
    metadata_file = DownloadedArtifact.metadata_filename(os.path.dirname(downloaded_artifact.path))
    with open(metadata_file) as fp:
        metadata = json.load(fp)
    metadata.pop("editable")
    metadata["version"] = DownloadedArtifact._LEGACY_METADATA_VERSION
    with open(metadata_file, "w") as fp:
        json.dump(metadata, fp)

    editable_artifact = download_manager.store(
        attr.evolve(local_project, editable=True), project_name
    )
    assert editable_artifact.editable
    assert 2 == len(
        download_manager.save_calls
    ), "Version 2 metadata must be rebuilt when the requested local project is editable."


def test_storage_version_upgrade(
    artifact,  # type: FileArtifact
    project_name,  # type: ProjectName
    download_manager,  # type: FakeDownloadManager
):
    # type: (...) -> None

    downloaded_artifact1 = download_manager.store(artifact, project_name)

    # If the storage metadata is not of the expected version, filename or just not present, we
    # expect the artifact to be re-stored afresh.
    files = tuple(
        os.path.join(root, f)
        for root, _, files in os.walk(os.path.dirname(downloaded_artifact1.path))
        for f in files
    )
    assert len(files) > 0, "We expect at least one metadata file."
    for f in files:
        os.unlink(f)

    downloaded_artifact2 = download_manager.store(artifact, project_name)
    assert downloaded_artifact1 == downloaded_artifact2
    assert 2 == len(
        download_manager.save_calls
    ), "Expected two save calls signalling a re-build of the artifact storage."
    assert 1 == len(set(download_manager.save_calls)), (
        "Expected each save call is with the same atomic directory work dir signalling a re-build "
        "of the same artifact storage."
    )


def test_storage_version_upgrade_holds_artifact_lock(
    artifact,  # type: FileArtifact
    project_name,  # type: ProjectName
    download_manager,  # type: FakeDownloadManager
    monkeypatch,  # type: Any
):
    # type: (...) -> None

    downloaded_artifact = download_manager.store(artifact, project_name)
    os.unlink(DownloadedArtifact.metadata_filename(os.path.dirname(downloaded_artifact.path)))

    lock_held = [False]
    original_locked = AtomicDirectory.locked

    @contextmanager
    def track_locked(atomic_dir, lock_style=None):
        # type: (AtomicDirectory, Any) -> Any
        with original_locked(atomic_dir, lock_style=lock_style):
            lock_held[0] = True
            try:
                yield
            finally:
                lock_held[0] = False

    removed = []

    def assert_locked_rmtree(path):
        # type: (str) -> None
        assert lock_held[0], "An invalid shared download must only be removed under its lock."
        removed.append(path)
        safe_rmtree(path)

    monkeypatch.setattr(AtomicDirectory, "locked", track_locked)
    monkeypatch.setattr(download_manager_module, "safe_rmtree", assert_locked_rmtree)

    assert downloaded_artifact == download_manager.store(artifact, project_name)
    assert [os.path.dirname(downloaded_artifact.path)] == removed


def test_storage_version_upgrade_rechecks_after_locking(
    artifact,  # type: FileArtifact
    project_name,  # type: ProjectName
    download_manager,  # type: FakeDownloadManager
    monkeypatch,  # type: Any
):
    # type: (...) -> None

    downloaded_artifact = download_manager.store(artifact, project_name)
    original_load = DownloadedArtifact.load
    load_calls = [0]

    def load_after_concurrent_repair(artifact_dir):
        # type: (str) -> DownloadedArtifact
        load_calls[0] += 1
        if load_calls[0] == 1:
            raise DownloadedArtifact.LoadError("simulated concurrent repair")
        return original_load(artifact_dir)

    def fail_if_removed(_path):
        # type: (str) -> None
        pytest.fail("A cache repaired before the locked re-check must not be removed.")

    monkeypatch.setattr(DownloadedArtifact, "load", load_after_concurrent_repair)
    monkeypatch.setattr(download_manager_module, "safe_rmtree", fail_if_removed)

    assert downloaded_artifact == download_manager.store(artifact, project_name)
    assert 2 == load_calls[0]
    assert 1 == len(download_manager.save_calls)


def test_storage_version_downgrade_v0(tmpdir):
    # type: (Any) -> None

    DownloadedArtifact.store(
        artifact_dir=str(tmpdir),
        filename="foo",
        legacy_fingerprint=Sha1Fingerprint("bar"),
        fingerprint=Sha256Fingerprint("baz"),
    )

    # We should always be emitting v0 metadata since versions of Pex that emitted that format did
    # not have an upgrade (downgrade) mechanism.
    with open(os.path.join(str(tmpdir), "sha1")) as fp:
        assert "bar" == fp.read()

    with open(DownloadedArtifact.metadata_filename(str(tmpdir))) as fp:
        assert (
            dict(
                algorithm="sha256",
                hexdigest="baz",
                filename="foo",
                subdirectory=None,
                editable=False,
                version=DownloadedArtifact._METADATA_VERSION,
            )
            == json.load(fp)
        )


def test_fingerprint_checking(
    expected_content,  # type: bytes
    artifact,  # type: FileArtifact
    project_name,  # type: ProjectName
    pex_root,  # type: str
):
    # type: (...) -> None

    # We expect un-verified artifacts to have their hashes checked against the expected (locked)
    # values.
    actual_content = b"unexpected content"
    download_manager = FakeDownloadManager(content=actual_content, pex_root=pex_root)
    expected_sha1_hash = hashing.Sha1(expected_content).hexdigest()
    assert Error(
        "Expected sha1 hash of {expected_hash} when downloading foo but hashed to "
        "{actual_hash}.".format(
            expected_hash=expected_sha1_hash, actual_hash=hashing.Sha1(actual_content).hexdigest()
        )
    ) == catch(download_manager.store, artifact, project_name)

    # But when the artifact hash is marked verified, no hash checking should occur.
    verified_artifact = attr.evolve(artifact, verified=True)
    expected_artifact_dir = CacheDir.DOWNLOADS.path(expected_sha1_hash, pex_root=pex_root)
    downloaded_artifact = download_manager.store(verified_artifact, project_name)
    assert (
        DownloadedArtifact(
            path=os.path.join(expected_artifact_dir, "foo-1.0.tar.gz"),
            fingerprint=hashing.Sha256(actual_content).hexdigest(),
        )
        == downloaded_artifact
    )
    assert downloaded_artifact == DownloadedArtifact.load(artifact_dir=expected_artifact_dir)
