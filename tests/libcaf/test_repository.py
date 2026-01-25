from pathlib import Path
from shutil import rmtree

from libcaf.constants import DEFAULT_BRANCH, HASH_LENGTH
from libcaf.plumbing import hash_object, load_commit, load_tree, open_content_for_reading
from libcaf.ref import RefError, SymRef, write_ref
from libcaf.repository import HashRef, Repository, RepositoryError, Tag, branch_ref
from pytest import raises


def test_init_with_custom_repo_dir(temp_repo_dir: Path) -> None:
    custom_repo_dir = '.custom_caf'
    repo = Repository(temp_repo_dir, custom_repo_dir)

    assert repo.repo_dir.name == custom_repo_dir
    assert str(repo.repo_dir) == custom_repo_dir

    repo.init()
    assert repo.exists()
    assert (temp_repo_dir / custom_repo_dir).exists()


def test_commit(temp_repo: Repository) -> None:
    temp_file = temp_repo.working_dir / 'test_file.txt'
    temp_file.write_text('This is a test file for commit.')

    author, message = 'John Doe', 'Initial commit'

    assert temp_repo.head_ref() == branch_ref(DEFAULT_BRANCH)

    commit_ref = temp_repo.commit_working_dir(author, message)
    commit = load_commit(temp_repo.objects_dir(), commit_ref)

    assert commit.author == author
    assert commit.message == message
    assert commit.tree_hash is not None

    # Check that HEAD remains pointing to the branch
    # and that the branch points to the commit
    assert temp_repo.head_ref() == branch_ref(DEFAULT_BRANCH)
    assert temp_repo.head_commit() == commit_ref

    commit_object = temp_repo.objects_dir() / commit_ref[:2] / commit_ref
    assert commit_object.exists()


def test_commit_with_parent(temp_repo: Repository) -> None:
    objects_dir = temp_repo.objects_dir()

    temp_file = temp_repo.working_dir / 'test_file.txt'
    temp_file.write_text('Initial commit content')
    temp_repo.save_file_content(temp_file)

    first_commit_ref = temp_repo.commit_working_dir('John Doe', 'First commit')
    first_commit = load_commit(objects_dir, first_commit_ref)

    assert first_commit_ref == hash_object(first_commit)
    assert temp_repo.head_commit() == first_commit_ref

    temp_file.write_text('Second commit content')
    temp_repo.save_file_content(temp_file)

    second_commit_ref = temp_repo.commit_working_dir('John Doe', 'Second commit')
    second_commit = load_commit(objects_dir, second_commit_ref)

    assert second_commit_ref == hash_object(second_commit)
    assert temp_repo.head_commit() == second_commit_ref

    assert second_commit.parent == first_commit_ref


def test_save_dir(temp_repo: Repository) -> None:
    test_dir = temp_repo.working_dir / 'test_dir'
    test_dir.mkdir()
    sub_dir = test_dir / 'sub_dir'
    sub_dir.mkdir()

    file1 = test_dir / 'file1.txt'
    file1.write_text('Content of file1')
    file2 = sub_dir / 'file2.txt'
    file2.write_text('Content of file2')
    file3 = sub_dir / 'file3.txt'
    file3.write_text('Content of file3')

    tree_ref = temp_repo.save_dir(test_dir)

    assert isinstance(tree_ref, HashRef)
    assert len(tree_ref) == HASH_LENGTH

    objects_dir = temp_repo.objects_dir()

    for file_path in [file1, file2, file3]:
        file_blob_hash = hash_object(temp_repo.save_file_content(file_path))
        assert (objects_dir / file_blob_hash[:2] / file_blob_hash).exists()

    assert (objects_dir / tree_ref[:2] / tree_ref).exists()


def test_head_log(temp_repo: Repository) -> None:
    temp_file = temp_repo.working_dir / 'commit_test.txt'

    temp_file.write_text('Initial commit')
    commit_ref1 = temp_repo.commit_working_dir('Author', 'First commit')

    temp_file.write_text('Second commit')
    commit_ref2 = temp_repo.commit_working_dir('Author', 'Second commit')

    assert [_.commit_ref for _ in temp_repo.log()] == [commit_ref2, commit_ref1]


def test_refs_directory_not_exists_raises_error(temp_repo: Repository) -> None:
    # Remove the refs directory to trigger the error condition
    refs_dir = temp_repo.refs_dir()
    rmtree(refs_dir)

    # This should raise RepositoryError because refs directory doesn't exist
    with raises(RepositoryError):
        temp_repo.refs()


def test_refs_directory_is_file_raises_error(temp_repo: Repository) -> None:
    refs_dir = temp_repo.refs_dir()

    # Remove refs directory if it exists and create a file with the same name
    rmtree(refs_dir)
    refs_dir.touch()

    # This should raise RepositoryError because refs directory is a file, not a directory
    with raises(RepositoryError):
        temp_repo.refs()


def test_resolve_ref_invalid_string_raises_error(temp_repo: Repository) -> None:
    with raises(RefError):
        temp_repo.resolve_ref('invalid_reference_string')

    with raises(RefError):
        temp_repo.resolve_ref('g' * HASH_LENGTH)  # 'g' is not a valid hex character

    with raises(RefError):
        temp_repo.resolve_ref('abc123')


def test_resolve_ref_invalid_type_raises_error(temp_repo: Repository) -> None:
    with raises(RefError):
        temp_repo.resolve_ref(123)

    with raises(RefError):
        temp_repo.resolve_ref([])

    with raises(RefError):
        temp_repo.resolve_ref({})


def test_update_ref_nonexistent_reference_raises_error(temp_repo: Repository) -> None:
    with raises(RepositoryError):
        temp_repo.update_ref('nonexistent_branch', HashRef('a' * 40))


def test_delete_repo_removes_repository(temp_repo: Repository) -> None:
    repo_path = temp_repo.repo_path()
    assert repo_path.exists()
    assert temp_repo.exists()

    temp_repo.delete_repo()

    assert not repo_path.exists()
    assert not temp_repo.exists()


def test_add_empty_branch_name_raises_error(temp_repo: Repository) -> None:
    with raises(ValueError, match='Branch name is required'):
        temp_repo.add_branch('')


def test_add_branch_already_exists_raises_error(temp_repo: Repository) -> None:
    with raises(RepositoryError):
        temp_repo.add_branch(DEFAULT_BRANCH)


def test_save_dir_invalid_path_raises_error(temp_repo: Repository) -> None:
    with raises(NotADirectoryError):
        temp_repo.save_dir(None)

    test_file = temp_repo.working_dir / 'test_file.txt'
    with raises(NotADirectoryError):
        temp_repo.save_dir(test_file)

    with raises(NotADirectoryError):
        temp_repo.save_dir(temp_repo.working_dir / 'non_existent_dir')


def test_delete_empty_branch_name_raises_error(temp_repo: Repository) -> None:
    with raises(ValueError, match='Branch name is required'):
        temp_repo.delete_branch('')


def test_delete_nonexistent_branch_name_raises_error(temp_repo: Repository) -> None:
    with raises(RepositoryError):
        temp_repo.delete_branch('nonexistent_branch')


def _create_sample_commit(repo: Repository) -> HashRef:
    sample_file = repo.working_dir / 'sample.txt'
    sample_file.write_text('sample content')
    repo.save_file_content(sample_file)
    return repo.commit_working_dir('Alice', 'sample commit')


def test_create_tag_and_list_tags(temp_repo: Repository) -> None:
    commit_ref = _create_sample_commit(temp_repo)

    created_tag = temp_repo.create_tag('v1.0', commit_ref)

    assert created_tag.name == 'v1.0'
    assert created_tag.target == commit_ref
    tags = temp_repo.list_tags()
    assert len(tags) == 1
    assert tags[0] == Tag('v1.0', commit_ref)


def test_create_tag_from_branch_name(temp_repo: Repository) -> None:
    commit_ref = _create_sample_commit(temp_repo)
    temp_repo.create_tag('v1.0', commit_ref)

    latest_commit = temp_repo.head_commit()
    assert latest_commit is not None
    tag = temp_repo.create_tag('stable', branch_ref(DEFAULT_BRANCH))
    assert tag.target == latest_commit


def test_create_tag_duplicate_name_raises_error(temp_repo: Repository) -> None:
    commit_ref = _create_sample_commit(temp_repo)
    temp_repo.create_tag('v1.0', commit_ref)

    with raises(RepositoryError):
        temp_repo.create_tag('v1.0', commit_ref)


def test_create_tag_invalid_target_raises_error(temp_repo: Repository) -> None:
    with raises(RepositoryError):
        temp_repo.create_tag('oops', 'deadbeef')


def test_delete_tag(temp_repo: Repository) -> None:
    commit_ref = _create_sample_commit(temp_repo)
    temp_repo.create_tag('v1.0', commit_ref)

    assert temp_repo.tag_exists('v1.0')
    temp_repo.delete_tag('v1.0')
    assert not temp_repo.tag_exists('v1.0')


def test_delete_missing_tag_raises_error(temp_repo: Repository) -> None:
    with raises(RepositoryError):
        temp_repo.delete_tag('missing')


def test_tag_exists_empty_name_raises_value_error(temp_repo: Repository) -> None:
    with raises(ValueError):
        temp_repo.tag_exists('')


def test_delete_last_branch_name_raises_error(temp_repo: Repository) -> None:
    with raises(RepositoryError):
        temp_repo.delete_branch('main')


def test_commit_working_dir_empty_author_or_message_raises_error(temp_repo: Repository) -> None:
    with raises(ValueError, match='Author is required'):
        temp_repo.commit_working_dir('', 'Valid message')

    with raises(ValueError, match='Author is required'):
        temp_repo.commit_working_dir(None, 'Valid message')  # type: ignore

    with raises(ValueError, match='Commit message is required'):
        temp_repo.commit_working_dir('Valid author', '')

    with raises(ValueError, match='Commit message is required'):
        temp_repo.commit_working_dir('Valid author', None)  # type: ignore

    with raises(ValueError, match='Author is required'):
        temp_repo.commit_working_dir('', '')


def test_log_corrupted_commit_raises_error(temp_repo: Repository) -> None:
    # First, create a valid commit
    temp_file = temp_repo.working_dir / 'test_file.txt'
    temp_file.write_text('Initial commit content')
    commit_ref = temp_repo.commit_working_dir('Author', 'Test commit')

    # Now corrupt the commit object by writing invalid data to it
    # Overwrite the commit object with invalid content
    objects_dir = temp_repo.objects_dir()
    commit_path = objects_dir / commit_ref[:2] / commit_ref
    commit_path.write_text('corrupted commit data')

    # Attempting to get the log should raise RepositoryError due to the corrupted commit
    with raises(RepositoryError):
        list(temp_repo.log())  # Convert generator to list to force evaluation


def test_diff_commits_corrupted_commit_raises_error(temp_repo: Repository) -> None:
    # First, create two valid commits
    temp_file = temp_repo.working_dir / 'test_file.txt'
    temp_file.write_text('Initial commit content')
    commit_ref1 = temp_repo.commit_working_dir('Author', 'First commit')

    temp_file.write_text('Second commit content')
    commit_ref2 = temp_repo.commit_working_dir('Author', 'Second commit')

    # Now corrupt the first commit object by writing invalid data to it
    # Overwrite the commit object with invalid content
    objects_dir = temp_repo.objects_dir()
    commit_path = objects_dir / commit_ref1[:2] / commit_ref1
    commit_path.write_text('corrupted commit data')

    # Attempting to diff commits should raise RepositoryError due to the corrupted commit
    with raises(RepositoryError):
        temp_repo.diff_commits(commit_ref1, commit_ref2)


def test_diff_commits_corrupted_tree_raises_error(temp_repo: Repository) -> None:
    # First, create two valid commits
    temp_file = temp_repo.working_dir / 'test_file.txt'
    temp_file.write_text('Initial commit content')
    commit_ref1 = temp_repo.commit_working_dir('Author', 'First commit')

    temp_file.write_text('Second commit content')
    commit_ref2 = temp_repo.commit_working_dir('Author', 'Second commit')

    # Load the commits to get their tree hashes
    objects_dir = temp_repo.objects_dir()
    commit1 = load_commit(objects_dir, commit_ref1)

    # Corrupt the tree object of the first commit
    # Overwrite the tree object with invalid content
    tree_hash = commit1.tree_hash
    tree_path = objects_dir / tree_hash[:2] / tree_hash
    tree_path.write_text('corrupted tree data')

    # Attempting to diff commits should raise RepositoryError due to the corrupted tree
    with raises(RepositoryError):
        temp_repo.diff_commits(commit_ref1, commit_ref2)


def test_diff_commits_corrupted_subtree_raises_error(temp_repo: Repository) -> None:
    # Create a directory structure with subdirectories to trigger recursive tree comparison
    test_dir = temp_repo.working_dir / 'test_dir'
    test_dir.mkdir()
    sub_dir = test_dir / 'sub_dir'
    sub_dir.mkdir()

    # Create files in both the main directory and subdirectory
    main_file = temp_repo.working_dir / 'main_file.txt'
    main_file.write_text('Main file content')
    sub_file = sub_dir / 'sub_file.txt'
    sub_file.write_text('Sub file content - version 1')

    # Create first commit
    commit_ref1 = temp_repo.commit_working_dir('Author', 'First commit with subdirectory')

    # Modify the file in the subdirectory to create a different tree
    sub_file.write_text('Sub file content - version 2')

    # Create second commit
    commit_ref2 = temp_repo.commit_working_dir('Author', 'Second commit with modified subdirectory')

    # Load the first commit to access its tree structure
    objects_dir = temp_repo.objects_dir()
    commit1 = load_commit(objects_dir, commit_ref1)

    # Load the root tree to find the subdirectory tree hash
    root_tree = load_tree(objects_dir, commit1.tree_hash)

    # Find the subdirectory tree record
    subdir_tree_hash: str | None = None
    for record in root_tree.records.values():
        if record.name == 'test_dir':
            # Load the test_dir tree to get its subdirectory
            test_dir_tree = load_tree(objects_dir, record.hash)
            for sub_record in test_dir_tree.records.values():
                if sub_record.name == 'sub_dir':
                    subdir_tree_hash = sub_record.hash
                    break
            break

    assert subdir_tree_hash is not None, 'Subdirectory tree hash should not be None'

    # Corrupt the subdirectory tree object
    tree_path = objects_dir / subdir_tree_hash[:2] / subdir_tree_hash
    tree_path.write_text('corrupted subtree data')

    # Attempting to diff commits should raise RepositoryError due to the corrupted subtree
    with raises(RepositoryError):
        temp_repo.diff_commits(commit_ref1, commit_ref2)


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


def _read_blob_text(repo: Repository, blob_hash: str) -> str:
    with open_content_for_reading(repo.objects_dir(), blob_hash) as handle:
        return handle.read().decode('utf-8')


def test_merge_commits_non_conflicting(temp_repo: Repository) -> None:
    base_file = temp_repo.working_dir / 'file_a.txt'
    base_file.write_text('base')
    base_commit = temp_repo.commit_working_dir('Author', 'Base commit')

    temp_repo.add_branch('feature')
    temp_repo.update_ref('heads/feature', base_commit)
    write_ref(temp_repo.head_file(), branch_ref('feature'))
    # TODO we should use update ref. Where is another example of update ref

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


def test_head_ref_missing_head_file_raises_error(temp_repo: Repository) -> None:
    # Remove the HEAD file to trigger the error condition
    temp_repo.head_file().unlink()

    # Attempting to get head_ref should raise RepositoryError due to missing HEAD file
    with raises(RepositoryError):
        temp_repo.head_ref()


def test_head_commit_with_symbolic_ref_returns_hash_ref(temp_repo: Repository) -> None:
    # First, create a commit so we have something to point to
    temp_file = temp_repo.working_dir / 'test_file.txt'
    temp_file.write_text('Test content')
    commit_ref = temp_repo.commit_working_dir('Author', 'Test commit')
    assert isinstance(commit_ref, HashRef)

    # Verify that HEAD is a symbolic reference pointing to a branch
    head_ref = temp_repo.head_ref()
    assert isinstance(head_ref, SymRef)

    # Update the HEAD to point to the commit we just created
    temp_repo.update_ref('heads/main', commit_ref)

    assert temp_repo.head_commit() == commit_ref


def test_merge_blob_text_triage_identical_ours_and_theirs(temp_repo: Repository) -> None:
    """Test hash triage when ours and theirs are identical - should skip merge."""
    from libcaf.repository import merge_blob_text
    
    # Create a blob
    test_file = temp_repo.working_dir / 'test.txt'
    test_file.write_text('same content')
    blob = temp_repo.save_file_content(test_file)
    
    # When ours and theirs are the same, should return that hash without conflict
    merged_hash, conflict = merge_blob_text(
        temp_repo.objects_dir(), 
        'different_base_hash',  # Base is different
        blob.hash,              # Ours
        blob.hash               # Theirs (same as ours)
    )
    
    assert merged_hash == blob.hash
    assert conflict is False


def test_merge_blob_text_triage_ours_unchanged(temp_repo: Repository) -> None:
    """Test hash triage when ours equals base - should take theirs."""
    from libcaf.repository import merge_blob_text
    
    # Create base/ours blob
    base_file = temp_repo.working_dir / 'base.txt'
    base_file.write_text('base content')
    base_blob = temp_repo.save_file_content(base_file)
    
    # Create theirs blob (different)
    theirs_file = temp_repo.working_dir / 'theirs.txt'
    theirs_file.write_text('theirs modified content')
    theirs_blob = temp_repo.save_file_content(theirs_file)
    
    # When ours == base, should return theirs without merge
    merged_hash, conflict = merge_blob_text(
        temp_repo.objects_dir(),
        base_blob.hash,   # Base
        base_blob.hash,   # Ours (same as base, unchanged)
        theirs_blob.hash  # Theirs (modified)
    )
    
    assert merged_hash == theirs_blob.hash
    assert conflict is False


def test_merge_blob_text_triage_theirs_unchanged(temp_repo: Repository) -> None:
    """Test hash triage when theirs equals base - should take ours."""
    from libcaf.repository import merge_blob_text
    
    # Create base/theirs blob
    base_file = temp_repo.working_dir / 'base.txt'
    base_file.write_text('base content')
    base_blob = temp_repo.save_file_content(base_file)
    
    # Create ours blob (different)
    ours_file = temp_repo.working_dir / 'ours.txt'
    ours_file.write_text('ours modified content')
    ours_blob = temp_repo.save_file_content(ours_file)
    
    # When theirs == base, should return ours without merge
    merged_hash, conflict = merge_blob_text(
        temp_repo.objects_dir(),
        base_blob.hash,  # Base
        ours_blob.hash,  # Ours (modified)
        base_blob.hash   # Theirs (same as base, unchanged)
    )
    
    assert merged_hash == ours_blob.hash
    assert conflict is False


def test_merge_blob_text_actual_merge_needed(temp_repo: Repository) -> None:
    """Test that actual merge happens when all three versions differ."""
    from libcaf.repository import merge_blob_text
    
    # Create base blob
    base_file = temp_repo.working_dir / 'base.txt'
    base_file.write_text('line 1\nline 2\nline 3\n')
    base_blob = temp_repo.save_file_content(base_file)
    
    # Create ours blob (modify line 1)
    ours_file = temp_repo.working_dir / 'ours.txt'
    ours_file.write_text('ours line 1\nline 2\nline 3\n')
    ours_blob = temp_repo.save_file_content(ours_file)
    
    # Create theirs blob (modify line 3)
    theirs_file = temp_repo.working_dir / 'theirs.txt'
    theirs_file.write_text('line 1\nline 2\ntheirs line 3\n')
    theirs_blob = temp_repo.save_file_content(theirs_file)
    
    # All three are different, should perform actual merge
    merged_hash, conflict = merge_blob_text(
        temp_repo.objects_dir(),
        base_blob.hash,
        ours_blob.hash,
        theirs_blob.hash
    )
    
    # Should successfully merge without conflict (changes on different lines)
    assert conflict is False
    assert merged_hash != base_blob.hash
    assert merged_hash != ours_blob.hash
    assert merged_hash != theirs_blob.hash
    
    # Verify merged content contains both changes
    merged_text = _read_blob_text(temp_repo, merged_hash)
    assert 'ours line 1' in merged_text
    assert 'theirs line 3' in merged_text


def test_merge_blob_text_actual_merge_with_conflict(temp_repo: Repository) -> None:
    """Test that conflicts are detected when both sides modify the same line."""
    from libcaf.repository import merge_blob_text
    
    # Create base blob
    base_file = temp_repo.working_dir / 'base.txt'
    base_file.write_text('line 1\n')
    base_blob = temp_repo.save_file_content(base_file)
    
    # Create ours blob (modify line 1)
    ours_file = temp_repo.working_dir / 'ours.txt'
    ours_file.write_text('ours modification\n')
    ours_blob = temp_repo.save_file_content(ours_file)
    
    # Create theirs blob (modify line 1 differently)
    theirs_file = temp_repo.working_dir / 'theirs.txt'
    theirs_file.write_text('theirs modification\n')
    theirs_blob = temp_repo.save_file_content(theirs_file)
    
    # Conflicting changes should be detected
    merged_hash, conflict = merge_blob_text(
        temp_repo.objects_dir(),
        base_blob.hash,
        ours_blob.hash,
        theirs_blob.hash
    )
    
    # Should detect conflict
    assert conflict is True
    
    # Verify merged content contains conflict markers
    merged_text = _read_blob_text(temp_repo, merged_hash)
    assert '<<<<<<<' in merged_text
    assert '=======' in merged_text
    assert '>>>>>>>' in merged_text


def test_merge_blob_text_no_base_identical_content(temp_repo: Repository) -> None:
    """Test 2-way merge when no base exists but content is identical."""
    from libcaf.repository import merge_blob_text
    
    # Create identical content in both branches
    content_file = temp_repo.working_dir / 'content.txt'
    content_file.write_text('same content added in both branches')
    blob = temp_repo.save_file_content(content_file)
    
    # No base, but ours and theirs are identical
    merged_hash, conflict = merge_blob_text(
        temp_repo.objects_dir(),
        None,       # No base (file didn't exist in common ancestor)
        blob.hash,  # Ours
        blob.hash   # Theirs (same)
    )
    
    assert merged_hash == blob.hash
    assert conflict is False


def test_merge_blob_text_no_base_different_content(temp_repo: Repository) -> None:
    """Test 2-way merge when no base exists and content differs."""
    from libcaf.repository import merge_blob_text
    
    # Create different content in both branches
    ours_file = temp_repo.working_dir / 'ours.txt'
    ours_file.write_text('ours added content')
    ours_blob = temp_repo.save_file_content(ours_file)
    
    theirs_file = temp_repo.working_dir / 'theirs.txt'
    theirs_file.write_text('theirs added content')
    theirs_blob = temp_repo.save_file_content(theirs_file)
    
    # No base, different content
    merged_hash, conflict = merge_blob_text(
        temp_repo.objects_dir(),
        None,            # No base
        ours_blob.hash,  # Ours
        theirs_blob.hash # Theirs (different)
    )
    
    # With no base and different content, merge3 treats everything as added
    # This will create a merged result with conflict markers
    merged_text = _read_blob_text(temp_repo, merged_hash)
    assert '<<<<<<<' in merged_text or 'ours added content' in merged_text
    assert 'theirs added content' in merged_text


# Tests for mmap-based merge implementation

def test_build_line_offsets(temp_repo: Repository) -> None:
    """Test that build_line_offsets correctly identifies line start positions."""
    import mmap
    from libcaf.repository import build_line_offsets
    
    # Create a file with known line positions
    test_file = temp_repo.working_dir / 'test_offsets.txt'
    content = 'line1\nline2\nline3\n'
    test_file.write_bytes(content.encode('utf-8'))
    
    with open(test_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offsets = build_line_offsets(mm)
        mm.close()
    
    # Should have 4 offsets: start of line1 (0), line2 (6), line3 (12), and after last newline (18)
    assert len(offsets) == 4
    assert offsets[0] == 0    # 'line1\n' starts at 0
    assert offsets[1] == 6    # 'line2\n' starts at 6
    assert offsets[2] == 12   # 'line3\n' starts at 12
    assert offsets[3] == 18   # Position after last newline


def test_build_line_offsets_no_trailing_newline(temp_repo: Repository) -> None:
    """Test line offsets for a file without trailing newline."""
    import mmap
    from libcaf.repository import build_line_offsets
    
    test_file = temp_repo.working_dir / 'test_no_newline.txt'
    content = 'line1\nline2'  # No trailing newline
    test_file.write_bytes(content.encode('utf-8'))
    
    with open(test_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offsets = build_line_offsets(mm)
        mm.close()
    
    # Should have 2 offsets: start of line1 (0) and start of line2 (6)
    assert len(offsets) == 2
    assert offsets[0] == 0
    assert offsets[1] == 6


def test_mmap_line_view_access(temp_repo: Repository) -> None:
    """Test that MMapLineView provides correct lazy line access."""
    import mmap
    from libcaf.repository import build_line_offsets, MMapLineView
    
    test_file = temp_repo.working_dir / 'test_view.txt'
    content = 'first line\nsecond line\nthird line\n'
    test_file.write_bytes(content.encode('utf-8'))
    
    with open(test_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offsets = build_line_offsets(mm)
        view = MMapLineView(mm, offsets)
        
        # Test length
        assert len(view) == 4  # 3 lines plus position after last newline
        
        # Test individual line access
        assert view[0] == 'first line\n'
        assert view[1] == 'second line\n'
        assert view[2] == 'third line\n'
        
        mm.close()


def test_mmap_line_view_out_of_range(temp_repo: Repository) -> None:
    """Test that MMapLineView raises IndexError for out of range access."""
    import mmap
    from libcaf.repository import build_line_offsets, MMapLineView
    
    test_file = temp_repo.working_dir / 'test_range.txt'
    test_file.write_bytes(b'one line\n')
    
    with open(test_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offsets = build_line_offsets(mm)
        view = MMapLineView(mm, offsets)
        
        with raises(IndexError):
            _ = view[100]
        
        with raises(IndexError):
            _ = view[-1]
        
        mm.close()


def test_incremental_merge_writer(temp_repo: Repository) -> None:
    """Test that IncrementalMergeWriter produces correct output and hash."""
    from libcaf.repository import IncrementalMergeWriter
    from libcaf.plumbing import hash_string
    
    writer = IncrementalMergeWriter(temp_repo.objects_dir())
    
    lines = ['line 1\n', 'line 2\n', 'line 3\n']
    for line in lines:
        writer.write_line(line)
    
    result_hash = writer.finalize()
    
    # Verify the hash matches what we expect
    expected_content = ''.join(lines)
    expected_hash = hash_string(expected_content)
    assert result_hash == expected_hash
    
    # Verify the content was saved correctly
    saved_content = _read_blob_text(temp_repo, result_hash)
    assert saved_content == expected_content


def test_mmap_merge_with_large_file(temp_repo: Repository) -> None:
    """Test mmap merge works correctly with larger files."""
    from libcaf.repository import merge_blob_text
    
    # Create a base file with many lines
    base_lines = [f'line {i}\n' for i in range(100)]
    base_file = temp_repo.working_dir / 'large_base.txt'
    base_file.write_text(''.join(base_lines))
    base_blob = temp_repo.save_file_content(base_file)
    
    # Create ours with modification at the beginning
    ours_lines = ['OURS CHANGE\n'] + base_lines[1:]
    ours_file = temp_repo.working_dir / 'large_ours.txt'
    ours_file.write_text(''.join(ours_lines))
    ours_blob = temp_repo.save_file_content(ours_file)
    
    # Create theirs with modification at the end
    theirs_lines = base_lines[:-1] + ['THEIRS CHANGE\n']
    theirs_file = temp_repo.working_dir / 'large_theirs.txt'
    theirs_file.write_text(''.join(theirs_lines))
    theirs_blob = temp_repo.save_file_content(theirs_file)
    
    # Merge should succeed without conflict
    merged_hash, conflict = merge_blob_text(
        temp_repo.objects_dir(),
        base_blob.hash,
        ours_blob.hash,
        theirs_blob.hash
    )
    
    assert conflict is False
    
    # Verify merged content contains both changes
    merged_text = _read_blob_text(temp_repo, merged_hash)
    assert 'OURS CHANGE' in merged_text
    assert 'THEIRS CHANGE' in merged_text


def test_mmap_merge_empty_file_ours(temp_repo: Repository) -> None:
    """Test mmap merge when our version is empty."""
    from libcaf.repository import merge_blob_text
    
    # Create base with content
    base_file = temp_repo.working_dir / 'base.txt'
    base_file.write_text('base content\n')
    base_blob = temp_repo.save_file_content(base_file)
    
    # Create empty file for ours
    ours_file = temp_repo.working_dir / 'ours_empty.txt'
    ours_file.write_text('')
    ours_blob = temp_repo.save_file_content(ours_file)
    
    # Theirs has same content as base
    # This should take our version (the deletion) since theirs == base
    merged_hash, conflict = merge_blob_text(
        temp_repo.objects_dir(),
        base_blob.hash,
        ours_blob.hash,
        base_blob.hash  # theirs same as base
    )
    
    # Our empty file should be chosen since theirs didn't change
    assert conflict is False
    assert merged_hash == ours_blob.hash


def test_mmap_merge_utf8_content(temp_repo: Repository) -> None:
    """Test mmap merge handles UTF-8 content correctly."""
    from libcaf.repository import merge_blob_text
    
    # Create files with UTF-8 content
    base_file = temp_repo.working_dir / 'utf8_base.txt'
    base_file.write_text('Hello 世界\nBonjour 🌍\n', encoding='utf-8')
    base_blob = temp_repo.save_file_content(base_file)
    
    # Ours modifies first line
    ours_file = temp_repo.working_dir / 'utf8_ours.txt'
    ours_file.write_text('Hola 世界\nBonjour 🌍\n', encoding='utf-8')
    ours_blob = temp_repo.save_file_content(ours_file)
    
    # Theirs modifies second line
    theirs_file = temp_repo.working_dir / 'utf8_theirs.txt'
    theirs_file.write_text('Hello 世界\nCiao 🌍\n', encoding='utf-8')
    theirs_blob = temp_repo.save_file_content(theirs_file)
    
    # Should merge without conflict (different lines modified)
    merged_hash, conflict = merge_blob_text(
        temp_repo.objects_dir(),
        base_blob.hash,
        ours_blob.hash,
        theirs_blob.hash
    )
    
    assert conflict is False
    
    merged_text = _read_blob_text(temp_repo, merged_hash)
    assert 'Hola 世界' in merged_text
    assert 'Ciao 🌍' in merged_text