"""libcaf repository management."""

import array
import mmap
import os
import shutil
from collections import deque
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Concatenate

from merge3 import Merge3

from . import Blob, Commit, Tree, TreeRecord, TreeRecordType
from .constants import (DEFAULT_BRANCH, DEFAULT_REPO_DIR, HASH_CHARSET, HASH_LENGTH, HEADS_DIR, HEAD_FILE,
                        OBJECTS_SUBDIR, REFS_DIR, TAGS_DIR)
from .plumbing import (get_content_path, hash_object, hash_string, load_commit, load_tree, open_content_for_reading,
                       open_content_for_writing, save_commit, save_file_content, save_tree)
from .ref import HashRef, Ref, RefError, SymRef, read_ref, write_ref


class RepositoryError(Exception):
    """Exception raised for repository-related errors."""


class RepositoryNotFoundError(RepositoryError):
    """Exception raised when a repository is not found."""


@dataclass
class Diff:
    """A class representing a difference between two tree records."""

    record: TreeRecord
    parent: 'Diff | None'
    children: list['Diff']


@dataclass
class AddedDiff(Diff):
    """An added tree record diff as part of a commit."""


@dataclass
class RemovedDiff(Diff):
    """A removed tree record diff as part of a commit."""


@dataclass
class ModifiedDiff(Diff):
    """A modified tree record diff as part of a commit."""


@dataclass
class MovedToDiff(Diff):
    """A tree record diff that has been moved elsewhere as part of a commit."""

    moved_to: 'MovedFromDiff | None'


@dataclass
class MovedFromDiff(Diff):
    """A tree record diff that has been moved from elsewhere as part of a commit."""

    moved_from: MovedToDiff | None


@dataclass
class LogEntry:
    """A class representing a log entry for a branch or commit history."""

    commit_ref: HashRef
    commit: Commit


@dataclass
class Tag:
    """Represents an immutable label that points to a commit."""

    name: str
    target: HashRef


@dataclass
class MergeResult:
    """Represents the output of a 3-way merge."""

    tree_hash: HashRef
    conflicts: list[str]


class Repository:
    """Represents a libcaf repository.

    This class provides methods to initialize a repository, manage branches,
    commit changes, and perform various operations on the repository."""

    def __init__(self, working_dir: Path | str, repo_dir: Path | str | None = None) -> None:
        """Initialize a Repository instance. The repository is not created on disk until `init()` is called.

        :param working_dir: The working directory where the repository will be located.
        :param repo_dir: The name of the repository directory within the working directory. Defaults to '.caf'."""
        self.working_dir = Path(working_dir)

        if repo_dir is None:
            self.repo_dir = Path(DEFAULT_REPO_DIR)
        else:
            self.repo_dir = Path(repo_dir)

    def init(self, default_branch: str = DEFAULT_BRANCH) -> None:
        """Initialize a new CAF repository in the working directory.

        :param default_branch: The name of the default branch to create. Defaults to 'main'.
        :raises RepositoryError: If the repository already exists or if the working directory is invalid."""
        self.repo_path().mkdir(parents=True)
        self.objects_dir().mkdir()

        heads_dir = self.heads_dir()
        heads_dir.mkdir(parents=True)
        self.tags_dir().mkdir(parents=True)

        self.add_branch(default_branch)

        write_ref(self.head_file(), branch_ref(default_branch))

    def exists(self) -> bool:
        """Check if the repository exists in the working directory.

        :return: True if the repository exists, False otherwise."""
        return self.repo_path().exists()

    def repo_path(self) -> Path:
        """Get the path to the repository directory.

        :return: The path to the repository directory."""
        return self.working_dir / self.repo_dir

    def objects_dir(self) -> Path:
        """Get the path to the objects directory within the repository.

        :return: The path to the objects directory."""
        return self.repo_path() / OBJECTS_SUBDIR

    def refs_dir(self) -> Path:
        """Get the path to the refs directory within the repository.

        :return: The path to the refs directory."""
        return self.repo_path() / REFS_DIR

    def heads_dir(self) -> Path:
        """Get the path to the heads directory within the repository.

        :return: The path to the heads directory."""
        return self.refs_dir() / HEADS_DIR

    def tags_dir(self) -> Path:
        """Get the path to the tags directory within the repository."""
        return self.refs_dir() / TAGS_DIR

    @staticmethod
    def requires_repo[**P, R](func: Callable[Concatenate['Repository', P], R]) -> \
            Callable[Concatenate['Repository', P], R]:
        """Decorate a Repository method to ensure that the repository exists before executing the method.

        :param func: The method to decorate.
        :return: A wrapper function that checks for the repository's existence."""

        @wraps(func)
        def _verify_repo(self: 'Repository', *args: P.args, **kwargs: P.kwargs) -> R:
            if not self.exists():
                msg = f'Repository not initialized at {self.repo_path()}'
                raise RepositoryNotFoundError(msg)

            return func(self, *args, **kwargs)

        return _verify_repo

    @requires_repo
    def head_ref(self) -> Ref | None:
        """Get the current HEAD reference of the repository.

        :return: The current HEAD reference, which can be a HashRef or SymRef.
        :raises RepositoryError: If the HEAD ref file does not exist.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        head_file = self.head_file()
        if not head_file.exists():
            msg = 'HEAD ref file does not exist'
            raise RepositoryError(msg)

        return read_ref(head_file)

    @requires_repo
    def head_commit(self) -> HashRef | None:
        """Return a ref to the current commit reference of the HEAD.

        :return: The current commit reference, or None if HEAD is not a commit.
        :raises RepositoryError: If the HEAD ref file does not exist.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        # If HEAD is a symbolic reference, resolve it to a hash
        resolved_ref = self.resolve_ref(self.head_ref())
        if resolved_ref:
            return resolved_ref
        return None

    @requires_repo
    def refs(self) -> list[SymRef]:
        """Get a list of all symbolic references in the repository.

        :return: A list of SymRef objects representing the symbolic references.
        :raises RepositoryError: If the refs directory does not exist or is not a directory.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        refs_dir = self.refs_dir()
        if not refs_dir.exists() or not refs_dir.is_dir():
            msg = f'Refs directory does not exist or is not a directory: {refs_dir}'
            raise RepositoryError(msg)

        refs: list[SymRef] = [SymRef(ref_file.name) for ref_file in refs_dir.rglob('*')
                              if ref_file.is_file()]

        return refs

    @requires_repo
    def resolve_ref(self, ref: Ref | str | None) -> HashRef | None:
        """Resolve a reference to a HashRef, following symbolic references if necessary.

        :param ref: The reference to resolve. This can be a HashRef, SymRef, or a string.
        :return: The resolved HashRef or None if the reference does not exist.
        :raises RefError: If the reference is invalid or cannot be resolved.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        match ref:
            case HashRef():
                return ref
            case SymRef(ref):
                if ref.upper() == 'HEAD':
                    return self.resolve_ref(self.head_ref())

                ref = read_ref(self.refs_dir() / ref)
                return self.resolve_ref(ref)
            case str():
                # Try to figure out what kind of ref it is by looking at the list of refs
                # in the refs directory
                if ref.upper() == 'HEAD' or ref in self.refs():
                    return self.resolve_ref(SymRef(ref))
                if len(ref) == HASH_LENGTH and all(c in HASH_CHARSET for c in ref):
                    return HashRef(ref)

                msg = f'Invalid reference: {ref}'
                raise RefError(msg)
            case None:
                return None
            case _:
                msg = f'Invalid reference type: {type(ref)}'
                raise RefError(msg)

    @requires_repo
    def update_ref(self, ref_name: str, new_ref: Ref) -> None:
        """Update a symbolic reference in the repository.

        :param ref_name: The name of the symbolic reference to update.
        :param new_ref: The new reference value to set.
        :raises RepositoryError: If the reference does not exist.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        ref_path = self.refs_dir() / ref_name

        if not ref_path.exists():
            msg = f'Reference "{ref_name}" does not exist.'
            raise RepositoryError(msg)

        write_ref(ref_path, new_ref)

    @requires_repo
    def delete_repo(self) -> None:
        """Delete the entire repository, including all objects and refs.

        :raises RepositoryNotFoundError: If the repository does not exist."""
        shutil.rmtree(self.repo_path())

    @requires_repo
    def save_file_content(self, file: Path) -> Blob:
        """Save the content of a file to the repository.

        :param file: The path to the file to save.
        :return: A Blob object representing the saved file content.
        :raises ValueError: If the file does not exist.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        return save_file_content(self.objects_dir(), file)

    @requires_repo
    def add_branch(self, branch: str) -> None:
        """Add a new branch to the repository, initialized to be an empty reference.

        :param branch: The name of the branch to add.
        :raises ValueError: If the branch name is empty.
        :raises RepositoryError: If the branch already exists.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        if not branch:
            msg = 'Branch name is required'
            raise ValueError(msg)
        if self.branch_exists(SymRef(branch)):
            msg = f'Branch "{branch}" already exists'
            raise RepositoryError(msg)

        (self.heads_dir() / branch).touch()

    @requires_repo
    def create_tag(self, tag_name: str, target: Ref | str) -> Tag:
        """Create a new tag that points to the given target commit.

        :param tag_name: The name of the tag to create.
        :param target: The reference (commit hash, branch, or tag) the new tag should point to.
        :return: The created Tag.
        :raises ValueError: If the tag name is empty.
        :raises RepositoryError: If the tag already exists or the target cannot be resolved.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        if not tag_name:
            msg = 'Tag name is required'
            raise ValueError(msg)

        tag_path = self.tags_dir() / tag_name
        if tag_path.exists():
            msg = f'Tag "{tag_name}" already exists'
            raise RepositoryError(msg)

        try:
            resolved_target = self.resolve_ref(target)
        except RefError as exc:
            msg = f'Cannot resolve target \"{target}\" for tag \"{tag_name}\"'
            raise RepositoryError(msg) from exc
        if resolved_target is None:
            msg = f'Cannot resolve target "{target}" for tag "{tag_name}"'
            raise RepositoryError(msg)

        try:
            load_commit(self.objects_dir(), resolved_target)
        except Exception as exc:
            msg = f'Cannot create tag \"{tag_name}\" because commit {resolved_target} does not exist'
            raise RepositoryError(msg) from exc

        tag_path.parent.mkdir(parents=True, exist_ok=True)
        write_ref(tag_path, resolved_target)

        return Tag(tag_name, resolved_target)

    @requires_repo
    def delete_tag(self, tag_name: str) -> None:
        """Delete a tag from the repository."""
        if not tag_name:
            msg = 'Tag name is required'
            raise ValueError(msg)

        tag_path = self.tags_dir() / tag_name
        if not tag_path.exists():
            msg = f'Tag "{tag_name}" does not exist.'
            raise RepositoryError(msg)

        tag_path.unlink()

    @requires_repo
    def list_tags(self) -> list[Tag]:
        """Return all tags sorted by name."""
        tags_dir = self.tags_dir()
        if not tags_dir.exists():
            return []

        tags: list[Tag] = []
        for tag_file in tags_dir.iterdir():
            if not tag_file.is_file():
                continue

            tag_ref_value = read_ref(tag_file)
            if not isinstance(tag_ref_value, HashRef):
                msg = f'Invalid tag reference stored in {tag_file}'
                raise RepositoryError(msg)

            tags.append(Tag(tag_file.name, tag_ref_value))

        tags.sort(key=lambda tag: tag.name)
        return tags

    @requires_repo
    def tag_exists(self, tag_name: str) -> bool:
        """Check whether a tag with the given name exists."""
        if not tag_name:
            msg = 'Tag name is required'
            raise ValueError(msg)

        return (self.tags_dir() / tag_name).exists()

    @requires_repo
    def delete_branch(self, branch: str) -> None:
        """Delete a branch from the repository.

        :param branch: The name of the branch to delete.
        :raises ValueError: If the branch name is empty.
        :raises RepositoryError: If the branch does not exist or if it is the last branch in the repository.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        if not branch:
            msg = 'Branch name is required'
            raise ValueError(msg)
        branch_path = self.heads_dir() / branch

        if not branch_path.exists():
            msg = f'Branch "{branch}" does not exist.'
            raise RepositoryError(msg)
        if len(self.branches()) == 1:
            msg = f'Cannot delete the last branch "{branch}".'
            raise RepositoryError(msg)

        branch_path.unlink()

    @requires_repo
    def branch_exists(self, branch_ref: Ref) -> bool:
        """Check if a branch exists in the repository.

        :param branch_ref: The reference to the branch to check.
        :return: True if the branch exists, False otherwise.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        return (self.heads_dir() / branch_ref).exists()

    @requires_repo
    def branches(self) -> list[str]:
        """Get a list of all branch names in the repository.

        :return: A list of branch names.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        return [x.name for x in self.heads_dir().iterdir() if x.is_file()]

    @requires_repo
    def save_dir(self, path: Path) -> HashRef:
        """Save the content of a directory to the repository.

        :param path: The path to the directory to save.
        :return: A HashRef object representing the saved directory tree object.
        :raises NotADirectoryError: If the path is not a directory.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        if not path or not path.is_dir():
            msg = f'{path} is not a directory'
            raise NotADirectoryError(msg)

        stack = deque([path])
        hashes: dict[Path, str] = {}

        while stack:
            current_path = stack.pop()
            tree_records: dict[str, TreeRecord] = {}

            for item in current_path.iterdir():
                if item.name == self.repo_dir.name:
                    continue
                if item.is_file():
                    blob = self.save_file_content(item)
                    tree_records[item.name] = TreeRecord(TreeRecordType.BLOB, blob.hash, item.name)
                elif item.is_dir():
                    if item in hashes:  # If the directory has already been processed, use its hash
                        subtree_hash = hashes[item]
                        tree_records[item.name] = TreeRecord(TreeRecordType.TREE, subtree_hash, item.name)
                    else:
                        stack.append(current_path)
                        stack.append(item)
                        break
            else:
                tree = Tree(tree_records)
                save_tree(self.objects_dir(), tree)
                hashes[current_path] = hash_object(tree)

        return HashRef(hashes[path])

    @requires_repo
    def commit_working_dir(self, author: str, message: str) -> HashRef:
        """Commit the current working directory to the repository.

        :param author: The name of the commit author.
        :param message: The commit message.
        :return: A HashRef object representing the commit reference.
        :raises ValueError: If the author or message is empty.
        :raises RepositoryError: If the commit process fails.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        if not author:
            msg = 'Author is required'
            raise ValueError(msg)
        if not message:
            msg = 'Commit message is required'
            raise ValueError(msg)

        # See if HEAD is a symbolic reference to a branch that we need to update
        # if the commit process is successful.
        # Otherwise, there is nothing to update and HEAD will continue to point
        # to the detached commit.
        # Either way the commit HEAD eventually resolves to becomes the parent of the new commit.
        head_ref = self.head_ref()
        branch = head_ref if isinstance(head_ref, SymRef) else None
        parent_commit_ref = self.head_commit()

        # Save the current working directory as a tree
        tree_hash = self.save_dir(self.working_dir)

        commit = Commit(tree_hash, author, message, int(datetime.now().timestamp()), parent_commit_ref)
        commit_ref = HashRef(hash_object(commit))

        save_commit(self.objects_dir(), commit)

        if branch:
            # Extract the relative path from the SymRef (e.g., 'heads/feature' from SymRef('heads/feature'))
            ref_path = str(branch)
            self.update_ref(ref_path, commit_ref)

        return commit_ref

    @requires_repo
    def log(self, tip: Ref | None = None) -> Generator[LogEntry, None, None]:
        """Generate a log of commits in the repository, starting from the specified tip.

        :param tip: The reference to the commit to start from. If None, defaults to the current HEAD.
        :return: A generator yielding LogEntry objects representing the commits in the log.
        :raises RepositoryError: If a commit cannot be loaded.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        tip = tip or self.head_ref()
        current_hash = self.resolve_ref(tip)

        try:
            while current_hash:
                commit = load_commit(self.objects_dir(), current_hash)
                yield LogEntry(HashRef(current_hash), commit)

                current_hash = HashRef(commit.parent) if commit.parent else None
        except Exception as e:
            msg = f'Error loading commit {current_hash}'
            raise RepositoryError(msg) from e

    @requires_repo
    def diff_commits(self, commit_ref1: Ref | None = None, commit_ref2: Ref | None = None) -> Sequence[Diff]:
        """Generate a diff between two commits in the repository.

        :param commit_ref1: The reference to the first commit. If None, defaults to the current HEAD.
        :param commit_ref2: The reference to the second commit. If None, defaults to the current HEAD.
        :return: A list of Diff objects representing the differences between the two commits.
        :raises RepositoryError: If a commit or tree cannot be loaded.
        :raises RepositoryNotFoundError: If the repository does not exist."""
        if commit_ref1 is None:
            commit_ref1 = self.head_ref()
        if commit_ref2 is None:
            commit_ref2 = self.head_ref()

        try:
            commit_hash1 = self.resolve_ref(commit_ref1)
            commit_hash2 = self.resolve_ref(commit_ref2)

            if commit_hash1 is None:
                msg = f'Cannot resolve reference {commit_ref1}'
                raise RefError(msg)
            if commit_hash2 is None:
                msg = f'Cannot resolve reference {commit_ref2}'
                raise RefError(msg)

            commit1 = load_commit(self.objects_dir(), commit_hash1)
            commit2 = load_commit(self.objects_dir(), commit_hash2)
        except Exception as e:
            msg = 'Error loading commit'
            raise RepositoryError(msg) from e

        if commit1.tree_hash == commit2.tree_hash:
            return []

        try:
            tree1 = load_tree(self.objects_dir(), commit1.tree_hash)
            tree2 = load_tree(self.objects_dir(), commit2.tree_hash)
        except Exception as e:
            msg = 'Error loading tree'
            raise RepositoryError(msg) from e

        top_level_diff = Diff(TreeRecord(TreeRecordType.TREE, '', ''), None, [])
        stack = [(tree1, tree2, top_level_diff)]

        potentially_added: dict[str, Diff] = {}
        potentially_removed: dict[str, Diff] = {}

        while stack:
            current_tree1, current_tree2, parent_diff = stack.pop()
            records1 = current_tree1.records if current_tree1 else {}
            records2 = current_tree2.records if current_tree2 else {}

            for name, record1 in records1.items():
                if name not in records2:
                    local_diff: Diff

                    # This name is no longer in the tree, so it was either moved or removed
                    # Have we seen this hash before as a potentially-added record?
                    if record1.hash in potentially_added:
                        added_diff = potentially_added[record1.hash]
                        del potentially_added[record1.hash]

                        local_diff = MovedToDiff(record1, parent_diff, [], None)
                        moved_from_diff = MovedFromDiff(added_diff.record, added_diff.parent, [], local_diff)
                        local_diff.moved_to = moved_from_diff

                        # Replace the original added diff with a moved-from diff
                        added_diff.parent.children = (
                            [_ if _.record.hash != record1.hash
                             else moved_from_diff
                             for _ in added_diff.parent.children])

                    else:
                        local_diff = RemovedDiff(record1, parent_diff, [])
                        potentially_removed[record1.hash] = local_diff

                    parent_diff.children.append(local_diff)
                else:
                    record2 = records2[name]

                    # This record is identical in both trees, so no diff is needed
                    if record1.hash == record2.hash:
                        continue

                    # If the record is a tree, we need to recursively compare the trees
                    if record1.type == TreeRecordType.TREE and record2.type == TreeRecordType.TREE:
                        subtree_diff = ModifiedDiff(record1, parent_diff, [])

                        try:
                            tree1 = load_tree(self.objects_dir(), record1.hash)
                            tree2 = load_tree(self.objects_dir(), record2.hash)
                        except Exception as e:
                            msg = 'Error loading subtree for diff'
                            raise RepositoryError(msg) from e

                        stack.append((tree1, tree2, subtree_diff))
                        parent_diff.children.append(subtree_diff)
                    else:
                        modified_diff = ModifiedDiff(record1, parent_diff, [])
                        parent_diff.children.append(modified_diff)

            for name, record2 in records2.items():
                if name not in records1:
                    # This name is in the new tree but not in the old tree, so it was either
                    # added or moved
                    # If we've already seen this hash, it was moved, so convert the original
                    # added diff to a moved diff
                    if record2.hash in potentially_removed:
                        removed_diff = potentially_removed[record2.hash]
                        del potentially_removed[record2.hash]

                        local_diff = MovedFromDiff(record2, parent_diff, [], None)
                        moved_to_diff = MovedToDiff(removed_diff.record, removed_diff.parent, [], local_diff)
                        local_diff.moved_from = moved_to_diff

                        # Create a new diff for the moved record
                        removed_diff.parent.children = (
                            [_ if _.record.hash != record2.hash
                             else moved_to_diff
                             for _ in removed_diff.parent.children])

                    else:
                        local_diff = AddedDiff(record2, parent_diff, [])
                        potentially_added[record2.hash] = local_diff

                    parent_diff.children.append(local_diff)

        # Sort the diffs to ensure a deterministic order.
        # top_level_diff.children.sort(key=lambda d: d.record.name)

        return top_level_diff.children

    @requires_repo
    def common_ancestor(self, commit_ref1: Ref | None = None, commit_ref2: Ref | None = None) -> HashRef | None:
        """Find the common ancestor of two commits, if one exists."""
        if commit_ref1 is None:
            commit_ref1 = self.head_ref()
        if commit_ref2 is None:
            commit_ref2 = self.head_ref()

        try:
            commit_hash1 = self.resolve_ref(commit_ref1)
            commit_hash2 = self.resolve_ref(commit_ref2)

            if commit_hash1 is None:
                raise RefError(f'Cannot resolve reference {commit_ref1}')
            if commit_hash2 is None:
                raise RefError(f'Cannot resolve reference {commit_ref2}')
                
        except Exception as e:
            msg = 'Error resolving commit references'
            raise RepositoryError(msg) from e

        return find_common_ancestor_core(self.objects_dir(), commit_hash1, commit_hash2)

    @requires_repo
    def merge_commits(self, commit_ref1: Ref | None = None, commit_ref2: Ref | None = None) -> MergeResult:
        """Perform a 3-way merge between two commits using their common ancestor."""
        if commit_ref1 is None:
            commit_ref1 = self.head_ref()
        if commit_ref2 is None:
            commit_ref2 = self.head_ref()

        try:
            commit_hash1 = self.resolve_ref(commit_ref1)
            commit_hash2 = self.resolve_ref(commit_ref2)

            if commit_hash1 is None:
                raise RefError(f'Cannot resolve reference {commit_ref1}')
            if commit_hash2 is None:
                raise RefError(f'Cannot resolve reference {commit_ref2}')

            ancestor_hash = find_common_ancestor_core(self.objects_dir(), commit_hash1, commit_hash2)
            if ancestor_hash is None:
                msg = 'No common ancestor found for merge'
                raise RepositoryError(msg)

            commit1 = load_commit(self.objects_dir(), commit_hash1)
            commit2 = load_commit(self.objects_dir(), commit_hash2)
            ancestor_commit = load_commit(self.objects_dir(), ancestor_hash)

            tree1 = load_tree(self.objects_dir(), commit1.tree_hash)
            tree2 = load_tree(self.objects_dir(), commit2.tree_hash)
            ancestor_tree = load_tree(self.objects_dir(), ancestor_commit.tree_hash)

        except Exception as e:
            msg = 'Error preparing commits for merge'
            raise RepositoryError(msg) from e

        conflicts: list[str] = []
        merged_tree_hash = merge_trees_core(
            self.objects_dir(),
            ancestor_tree,
            tree1,
            tree2,
            '',
            conflicts,
        )

        return MergeResult(merged_tree_hash, conflicts)
        

    def head_file(self) -> Path:
        """Get the path to the HEAD file within the repository.

        :return: The path to the HEAD file."""
        return self.repo_path() / HEAD_FILE


def branch_ref(branch: str) -> SymRef:
    """Create a symbolic reference for a branch name.

    :param branch: The name of the branch.
    :return: A SymRef object representing the branch reference."""
    return SymRef(f'{HEADS_DIR}/{branch}')


def tag_ref(tag: str) -> SymRef:
    """Create a symbolic reference for a tag name."""
    return SymRef(f'{TAGS_DIR}/{tag}')


def build_line_offsets(mm: mmap.mmap) -> array.array:
    """Build a compact array of line start byte positions.
    
    :param mm: Memory-mapped file object
    :return: Array of line start positions (first element is always 0)
    """
    offsets = array.array('Q')  # Unsigned long long
    offsets.append(0)
    
    pos = 0
    while (nl := mm.find(b'\n', pos)) != -1:
        offsets.append(nl + 1)  # Next line starts after the newline
        pos = nl + 1
    
    return offsets


class MMapLineView:
    """Provides lazy line-by-line access to a memory-mapped file.
    
    This class wraps a memory-mapped file and an array of line offsets
    to provide sequence-like access to lines without loading the entire
    file into memory. Lines are only decoded when accessed via __getitem__.
    
    Implements __len__ and __getitem__ to satisfy merge3's requirements
    for a sequence-like object.
    """
    
    def __init__(self, mm: mmap.mmap, offsets: array.array) -> None:
        """Initialize the line view.
        
        :param mm: Memory-mapped file object
        :param offsets: Array of line start byte positions
        """
        self.mm = mm
        self.offsets = offsets
    
    def __getitem__(self, idx: int) -> str:
        """Get a line by index, decoding from UTF-8.
        
        :param idx: Line index (0-based)
        :return: The line content as a string (including trailing newline if present)
        :raises IndexError: If idx is out of range
        """
        if idx < 0 or idx >= len(self.offsets):
            msg = f'Line index {idx} out of range'
            raise IndexError(msg)
        
        start = self.offsets[idx]
        # End is the start of the next line, or end of file
        end = self.offsets[idx + 1] if idx + 1 < len(self.offsets) else len(self.mm)
        return self.mm[start:end].decode('utf-8')
    
    def __len__(self) -> int:
        """Return the number of lines in the file.
        
        :return: Number of lines
        """
        return len(self.offsets)


def read_blob_text(objects_dir: str | Path, blob_hash: str) -> str:
    """Load blob content as UTF-8 text."""
    try:
        with open_content_for_reading(objects_dir, blob_hash) as handle:
            content = handle.read()
        return content.decode('utf-8')
    except Exception as e:
        msg = f'Error reading blob {blob_hash}'
        raise RepositoryError(msg) from e


def save_blob_text(objects_dir: str | Path, content: str) -> HashRef:
    """Save UTF-8 text content as a blob and return its hash."""
    try:
        blob_hash = HashRef(hash_string(content))
        with open_content_for_writing(objects_dir, blob_hash) as handle:
            handle.write(content.encode('utf-8'))
    except Exception as e:
        msg = 'Error saving merged blob content'
        raise RepositoryError(msg) from e

    return blob_hash


def _create_empty_line_view() -> MMapLineView:
    """Create an empty line view for files that don't exist (new files).
    
    :return: MMapLineView with empty content
    """
    # Create a minimal mmap-like object for empty content
    empty_mm = mmap.mmap(-1, 1)  # Anonymous mmap with 1 byte
    empty_mm.write(b'\x00')
    empty_mm.seek(0)
    empty_offsets = array.array('Q')
    # Return a view that will have 0 lines
    return MMapLineView(empty_mm, empty_offsets)


def _mmap_blob(objects_dir: Path, blob_hash: str | None) -> tuple[mmap.mmap | None, MMapLineView]:
    """Memory-map a blob file and create a line view.
    
    :param objects_dir: Directory containing blob objects
    :param blob_hash: Hash of the blob to map, or None for empty content
    :return: Tuple of (mmap object or None, MMapLineView)
    """
    if blob_hash is None:
        return None, _create_empty_line_view()
    
    file_handle = open_content_for_reading(objects_dir, blob_hash)
    file_size = os.fstat(file_handle.fileno()).st_size
    if file_size == 0:
        file_handle.close()
        return None, _create_empty_line_view()
    
    # Open and memory-map the file
    mm = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
    file_handle.close()  # File handle can be closed, mmap keeps the mapping
    
    offsets = build_line_offsets(mm)
    return mm, MMapLineView(mm, offsets)


def merge_blob_text(
    objects_dir: Path,
    base_hash: str | None,
    ours_hash: str | None,
    theirs_hash: str | None,
) -> tuple[HashRef, bool]:
    """Merge three versions of a blob using merge3.
    
    Uses hash-based triage to avoid unnecessary merges when possible.
    For actual merges, uses mmap for memory-efficient file reading and
    incremental writing for the merged output.
    
    All content is assumed to be UTF-8 encoded text.
    
    :param objects_dir: Directory containing the blob objects
    :param base_hash: Hash of the common ancestor blob (None if no common ancestor)
    :param ours_hash: Hash of our version of the blob
    :param theirs_hash: Hash of their version of the blob
    :return: Tuple of (merged_blob_hash, has_conflict)
    """
    
    # HASH-BASED TRIAGE - Compare hashes before reading any content
    # This avoids expensive file I/O in the most common cases
    
    # Case 1: ours and theirs are identical
    if ours_hash == theirs_hash:
        # Both branches have the same content, no merge needed
        return HashRef(ours_hash), False
    
    # Case 2: base exists and ours didn't change
    if base_hash and ours_hash == base_hash:
        # We didn't modify the file, take their version
        return HashRef(theirs_hash), False
    
    # Case 3: base exists and theirs didn't change
    if base_hash and theirs_hash == base_hash:
        # They didn't modify the file, take our version
        return HashRef(ours_hash), False
    
    # If we reach here, we need to actually perform a 3-way merge
    # Both sides modified the file differently
    
    # Memory-map all three files for efficient access
    base_mm = None
    ours_mm = None
    theirs_mm = None
    
    try:
        base_mm, base_view = _mmap_blob(objects_dir, base_hash)
        ours_mm, ours_view = _mmap_blob(objects_dir, ours_hash)
        theirs_mm, theirs_view = _mmap_blob(objects_dir, theirs_hash)
        
        # Perform 3-way merge using lazy line views
        merger = Merge3(base_view, ours_view, theirs_view)
        
        content_buffer: list[bytes] = []
        conflict = False
        for group in merger.merge_groups():
            if not group:
                continue
            
            group_type = group[0]
            
            if group_type == 'unchanged':
                # Lines unchanged from base - output them
                _, base_start, base_end, _, _, _, _ = group
                for i in range(base_start, base_end):
                    content_buffer.append(base_view[i].encode('utf-8'))
            
            elif group_type == 'same':
                # Lines changed but both sides made the same change
                _, _, _, a_start, a_end, _, _ = group
                for i in range(a_start, a_end):
                    content_buffer.append(ours_view[i].encode('utf-8'))
            
            elif group_type == 'a':
                # Only 'ours' side changed
                _, _, _, a_start, a_end, _, _ = group
                for i in range(a_start, a_end):
                    content_buffer.append(ours_view[i].encode('utf-8'))
            
            elif group_type == 'b':
                # Only 'theirs' side changed
                _, _, _, _, _, b_start, b_end = group
                for i in range(b_start, b_end):
                    content_buffer.append(theirs_view[i].encode('utf-8'))
            
            elif group_type == 'conflict':
                # Both sides changed differently - mark conflict and output both versions
                conflict = True
                _, _, _, a_start, a_end, b_start, b_end = group
                
                content_buffer.append(b'<<<<<<< ours\n')
                for i in range(a_start, a_end):
                    content_buffer.append(ours_view[i].encode('utf-8'))
                content_buffer.append(b'=======\n')
                for i in range(b_start, b_end):
                    content_buffer.append(theirs_view[i].encode('utf-8'))
                content_buffer.append(b'>>>>>>> theirs\n')
        
        # Finalize and get the hash
        content_bytes = b''.join(content_buffer)
        merged_hash = HashRef(hash_string(content_bytes))
        with open_content_for_writing(objects_dir, merged_hash) as handle:
            handle.write(content_bytes)
        
        return merged_hash, conflict
        
    finally:
        # Clean up mmap objects
        if base_mm is not None:
            base_mm.close()
        if ours_mm is not None:
            ours_mm.close()
        if theirs_mm is not None:
            theirs_mm.close()


def records_match(record1: TreeRecord | None, record2: TreeRecord | None) -> bool:
    if record1 is None or record2 is None:
        return False
    return record1.type == record2.type and record1.hash == record2.hash


def merge_trees_core(
    objects_dir: Path,
    base_tree: Tree | None,
    ours_tree: Tree | None,
    theirs_tree: Tree | None,
    path_prefix: str,
    conflicts: list[str],
) -> HashRef:
    """Merge trees iteratively using a stack to avoid unbounded recursion."""
    # Stack contains tuples of: (base_tree, ours_tree, theirs_tree, path_prefix, pending_names, merged_records)
    # pending_names is the sorted list of names still to process
    # merged_records accumulates the results for this tree level
    stack: list[tuple[Tree | None, Tree | None, Tree | None, str, list[str], dict[str, TreeRecord]]] = []
    
    # Cache of completed subtree hashes keyed by (base_hash, ours_hash, theirs_hash, path)
    completed_subtrees: dict[tuple[str | None, str | None, str | None, str], HashRef] = {}
    
    # Initialize with the root tree
    base_records = base_tree.records if base_tree else {}
    ours_records = ours_tree.records if ours_tree else {}
    theirs_records = theirs_tree.records if theirs_tree else {}
    all_names = sorted(set(base_records) | set(ours_records) | set(theirs_records))
    
    root_merged_records: dict[str, TreeRecord] = {}
    stack.append((base_tree, ours_tree, theirs_tree, path_prefix, all_names, root_merged_records))
    
    while stack:
        current_base, current_ours, current_theirs, current_path_prefix, pending_names, merged_records = stack.pop()
        
        base_records = current_base.records if current_base else {}
        ours_records = current_ours.records if current_ours else {}
        theirs_records = current_theirs.records if current_theirs else {}
        
        # Process all pending names for this tree level
        subtree_encountered = False
        while pending_names:
            name = pending_names[0]
            
            base_record = base_records.get(name)
            our_record = ours_records.get(name)
            theirs_record = theirs_records.get(name)
            
            path = f'{current_path_prefix}/{name}' if current_path_prefix else name
            
            # Case 1: Both sides match - use either one
            if records_match(our_record, theirs_record):
                merged_records[name] = our_record
                pending_names.pop(0)
                continue
            
            # Case 2: Ours matches base - take theirs (if it exists)
            if records_match(base_record, our_record):
                if theirs_record is not None:
                    merged_records[name] = TreeRecord(theirs_record.type, theirs_record.hash, name)
                pending_names.pop(0)
                continue
            
            # Case 3: Theirs matches base - take ours (if it exists)
            if records_match(base_record, theirs_record):
                if our_record is not None:
                    merged_records[name] = TreeRecord(our_record.type, our_record.hash, name)
                pending_names.pop(0)
                continue
            
            # Case 4: Base is None and only one side has the record - take that side
            exactly_one_side_added = (our_record is None) != (theirs_record is None)
            if base_record is None and exactly_one_side_added:
                chosen = our_record or theirs_record
                merged_records[name] = TreeRecord(chosen.type, chosen.hash, name)
                pending_names.pop(0)
                continue
            
            # Case 5: Both sides are trees - need to merge recursively
            if (
                our_record
                and theirs_record
                and our_record.type == TreeRecordType.TREE
                and theirs_record.type == TreeRecordType.TREE
            ):
                # Create a cache key for this subtree merge
                base_hash = base_record.hash if base_record and base_record.type == TreeRecordType.TREE else None
                cache_key = (base_hash, our_record.hash, theirs_record.hash, path)
                
                # Check if we've already computed this subtree merge
                if cache_key in completed_subtrees:
                    merged_records[name] = TreeRecord(TreeRecordType.TREE, completed_subtrees[cache_key], name)
                    pending_names.pop(0)
                    continue
                
                # Need to process subtree - push current state back onto stack
                stack.append((current_base, current_ours, current_theirs, current_path_prefix, pending_names, merged_records))
                
                # Load the subtrees
                base_subtree = (
                    load_tree(objects_dir, base_record.hash)
                    if base_record and base_record.type == TreeRecordType.TREE
                    else None
                )
                ours_subtree = load_tree(objects_dir, our_record.hash)
                theirs_subtree = load_tree(objects_dir, theirs_record.hash)
                
                # Push subtree processing onto stack
                subtree_names = sorted(
                    set(base_subtree.records if base_subtree else {}) |
                    set(ours_subtree.records if ours_subtree else {}) |
                    set(theirs_subtree.records if theirs_subtree else {})
                )
                subtree_merged_records: dict[str, TreeRecord] = {}
                stack.append((base_subtree, ours_subtree, theirs_subtree, path, subtree_names, subtree_merged_records))
                
                subtree_encountered = True
                break
            
            # Case 6: Both sides are blobs - merge the content
            if (
                our_record
                and theirs_record
                and our_record.type == TreeRecordType.BLOB
                and theirs_record.type == TreeRecordType.BLOB
            ):
                base_hash = base_record.hash if base_record and base_record.type == TreeRecordType.BLOB else None
                merged_hash, conflict = merge_blob_text(objects_dir, base_hash, our_record.hash, theirs_record.hash)
                if conflict:
                    conflicts.append(path)
                merged_records[name] = TreeRecord(TreeRecordType.BLOB, merged_hash, name)
                pending_names.pop(0)
                continue
            
            # Case 7: Conflict - choose one and mark as conflict
            chosen = our_record or theirs_record
            if chosen is not None:
                merged_records[name] = TreeRecord(chosen.type, chosen.hash, name)
            conflicts.append(path)
            pending_names.pop(0)
        
        # If we encountered a subtree, continue to process it
        if subtree_encountered:
            continue
        
        # All names processed for this tree level - save the merged tree
        merged_tree = Tree(merged_records)
        save_tree(objects_dir, merged_tree)
        tree_hash = hash_object(merged_tree)
        
        # If this was a subtree, cache it and update parent
        if stack:
            # Pop the parent frame to update it
            parent_base, parent_ours, parent_theirs, parent_path_prefix, parent_pending_names, parent_merged_records = stack.pop()
            
            # The first pending name in parent is the subtree we just processed
            if parent_pending_names:
                subtree_name = parent_pending_names[0]
                
                parent_base_records = parent_base.records if parent_base else {}
                parent_ours_records = parent_ours.records if parent_ours else {}
                parent_theirs_records = parent_theirs.records if parent_theirs else {}
                
                parent_base_record = parent_base_records.get(subtree_name)
                parent_our_record = parent_ours_records.get(subtree_name)
                parent_theirs_record = parent_theirs_records.get(subtree_name)
                
                # Cache this subtree result
                subtree_path = f'{parent_path_prefix}/{subtree_name}' if parent_path_prefix else subtree_name
                base_hash = parent_base_record.hash if parent_base_record and parent_base_record.type == TreeRecordType.TREE else None
                ours_hash = parent_our_record.hash if parent_our_record else None
                theirs_hash = parent_theirs_record.hash if parent_theirs_record else None
                cache_key = (base_hash, ours_hash, theirs_hash, subtree_path)
                completed_subtrees[cache_key] = tree_hash
                
                # Update parent's merged records
                parent_merged_records[subtree_name] = TreeRecord(TreeRecordType.TREE, tree_hash, subtree_name)
                parent_pending_names.pop(0)
            
            # Push parent back onto stack to continue processing
            stack.append((parent_base, parent_ours, parent_theirs, parent_path_prefix, parent_pending_names, parent_merged_records))
        else:
            # This was the root tree - return its hash
            return tree_hash
    
    # Should never reach here, but return a placeholder
    msg = "Unexpected end of merge_trees_core"
    raise RepositoryError(msg)

def find_common_ancestor_core(objects_dir: str, hash1: str, hash2: str) -> HashRef | None:
    """Helper function to run the ancestor search algorithm independent of the Repository class."""
    try:
        ancestors: set[HashRef] = set()
        current_hash = hash1
        while current_hash:
            ancestors.add(HashRef(current_hash))
            commit = load_commit(objects_dir, current_hash)
            parent = commit.parent
            current_hash = HashRef(parent) if parent else parent

        current_hash2 = hash2
        while current_hash2:
            if current_hash2 in ancestors:
                return HashRef(current_hash2)
            commit = load_commit(objects_dir, current_hash2)
            parent = commit.parent
            current_hash2 = HashRef(parent) if parent else parent
            
    except Exception as e:
        msg = 'Error loading commit during ancestor search'
        raise RepositoryError(msg) from e

    return None
