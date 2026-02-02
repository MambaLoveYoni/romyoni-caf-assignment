from pathlib import Path

from libcaf.constants import DEFAULT_BRANCH
from libcaf.plumbing import load_tree, open_content_for_reading
from libcaf.ref import write_ref
from libcaf.repository import Repository, RepositoryError, branch_ref
from pytest import raises


def _read_blob_text(repo: Repository, blob_hash: str) -> str:
    with open_content_for_reading(repo.objects_dir(), blob_hash) as handle:
        return handle.read().decode('utf-8')


def test_common_ancestor_linear_history(temp_repo: Repository) -> None:
    temp_file = temp_repo.working_dir / 'test_file.txt'
    temp_file.write_text('Initial commit content')
    base_commit = temp_repo.commit_working_dir('Author', 'Base commit')

    temp_file.write_text('Second commit content')
    tip_commit = temp_repo.commit_working_dir('Author', 'Second commit')

    assert temp_repo.common_ancestor(tip_commit, base_commit) == base_commit


def test_common_ancestor_branches(temp_repo: Repository) -> None:
    temp_file = temp_repo.working_dir / 'test_file.txt'
    temp_file.write_text('Base content')
    base_commit = temp_repo.commit_working_dir('Author', 'Base commit')

    temp_file.write_text('Main branch change')
    main_commit = temp_repo.commit_working_dir('Author', 'Main commit')

    temp_repo.add_branch('feature')
    temp_repo.update_ref('heads/feature', base_commit)
    write_ref(temp_repo.head_file(), branch_ref('feature'))

    temp_file.write_text('Feature branch change')
    feature_commit = temp_repo.commit_working_dir('Author', 'Feature commit')

    assert temp_repo.common_ancestor(main_commit, feature_commit) == base_commit
    assert temp_repo.common_ancestor(feature_commit, main_commit) == base_commit


def test_common_ancestor_no_common_root(temp_repo: Repository) -> None:
    temp_file = temp_repo.working_dir / 'test_file.txt'
    temp_file.write_text('Root A')
    root_a = temp_repo.commit_working_dir('Author', 'Root A')

    temp_repo.head_file().write_text('')
    temp_file.write_text('Root B')
    root_b = temp_repo.commit_working_dir('Author', 'Root B')

    assert temp_repo.common_ancestor(root_a, root_b) is None


def test_merge_commits_non_conflicting(temp_repo: Repository) -> None:
    base_file = temp_repo.working_dir / 'file_a.txt'
    base_file.write_text('base')
    base_commit = temp_repo.commit_working_dir('Author', 'Base commit')

    temp_repo.add_branch('feature')
    temp_repo.update_ref('heads/feature', base_commit)
    write_ref(temp_repo.head_file(), branch_ref('feature'))

    feature_file = temp_repo.working_dir / 'file_b.txt'
    feature_file.write_text('feature content')
    feature_commit = temp_repo.commit_working_dir('Author', 'Feature commit')

    feature_file.unlink()
    write_ref(temp_repo.head_file(), branch_ref(DEFAULT_BRANCH))

    base_file.write_text('main change')
    main_commit = temp_repo.commit_working_dir('Author', 'Main commit')

    merge_result = temp_repo.merge_commits(main_commit, feature_commit)
    assert merge_result.conflicts == []

    merged_tree = load_tree(temp_repo.objects_dir(), merge_result.tree_hash)
    assert 'file_a.txt' in merged_tree.records
    assert 'file_b.txt' in merged_tree.records

    file_a_hash = merged_tree.records['file_a.txt'].hash
    file_b_hash = merged_tree.records['file_b.txt'].hash
    assert _read_blob_text(temp_repo, file_a_hash) == 'main change'
    assert _read_blob_text(temp_repo, file_b_hash) == 'feature content'


def test_merge_commits_conflict_same_file(temp_repo: Repository) -> None:
    base_file = temp_repo.working_dir / 'file_a.txt'
    base_file.write_text('base')
    base_commit = temp_repo.commit_working_dir('Author', 'Base commit')

    temp_repo.add_branch('feature')
    temp_repo.update_ref('heads/feature', base_commit)
    write_ref(temp_repo.head_file(), branch_ref('feature'))

    base_file.write_text('feature change')
    feature_commit = temp_repo.commit_working_dir('Author', 'Feature commit')

    write_ref(temp_repo.head_file(), branch_ref(DEFAULT_BRANCH))

    base_file.write_text('main change')
    main_commit = temp_repo.commit_working_dir('Author', 'Main commit')

    merge_result = temp_repo.merge_commits(main_commit, feature_commit)
    assert 'file_a.txt' in merge_result.conflicts

    merged_tree = load_tree(temp_repo.objects_dir(), merge_result.tree_hash)
    merged_blob = merged_tree.records['file_a.txt'].hash
    merged_text = _read_blob_text(temp_repo, merged_blob)
    assert '<<<<<<<' in merged_text
    assert '=======' in merged_text
    assert '>>>>>>>' in merged_text


def test_merge_commits_no_common_ancestor_raises_error(temp_repo: Repository) -> None:
    temp_file = temp_repo.working_dir / 'test_file.txt'
    temp_file.write_text('Root A')
    root_a = temp_repo.commit_working_dir('Author', 'Root A')

    temp_repo.head_file().write_text('')
    temp_file.write_text('Root B')
    root_b = temp_repo.commit_working_dir('Author', 'Root B')

    with raises(RepositoryError):
        temp_repo.merge_commits(root_a, root_b)


def test_merge_commits_binary_file_conflict(temp_repo: Repository) -> None:
    binary_file = temp_repo.working_dir / 'image.bin'
    binary_file.write_bytes(b'\x00\x01\x02\x03\x04base binary data')
    base_commit = temp_repo.commit_working_dir('Author', 'Base commit with binary')

    temp_repo.add_branch('feature')
    temp_repo.update_ref('heads/feature', base_commit)
    write_ref(temp_repo.head_file(), branch_ref('feature'))

    binary_file.write_bytes(b'\x00\x01\x02\x03\x04feature binary data')
    feature_commit = temp_repo.commit_working_dir('Author', 'Feature modifies binary')

    write_ref(temp_repo.head_file(), branch_ref(DEFAULT_BRANCH))

    binary_file.write_bytes(b'\x00\x01\x02\x03\x04main binary data')
    main_commit = temp_repo.commit_working_dir('Author', 'Main modifies binary')

    merge_result = temp_repo.merge_commits(main_commit, feature_commit)
    assert 'image.bin' in merge_result.conflicts

    merged_tree = load_tree(temp_repo.objects_dir(), merge_result.tree_hash)
    merged_blob_hash = merged_tree.records['image.bin'].hash

    with open_content_for_reading(temp_repo.objects_dir(), merged_blob_hash) as handle:
        merged_content = handle.read()
    assert merged_content == b'\x00\x01\x02\x03\x04main binary data'
