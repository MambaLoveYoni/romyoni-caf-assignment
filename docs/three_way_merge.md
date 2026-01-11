# Three-Way Merge (merge3) in CAF

## What is a 3-way merge? (very simple)
Imagine you and a friend both start from the same photo (the "base"). You each make changes. A 3-way merge compares:
1) the original photo,
2) your edited photo,
3) your friend's edited photo.
If only one person changed something, the merge takes that change. If both changed the same part differently, it marks a conflict so a human can decide.

## What I implemented
- A new merge API in `libcaf/libcaf/repository.py` that performs a 3-way merge between two commits.
- A recursive tree merge that handles folders and files.
- A text merge for blobs using the `merge3` library.
- A small plumbing helper to hash string content (`hash_string`) so merged text can be stored as new blobs.

## Key code sections explained

### `MergeResult`
This small dataclass is the return value of the merge. It contains:
- `tree_hash`: the hash of the merged tree.
- `conflicts`: a list of file paths that had conflicts.

### `Repository.merge_commits`
This method is the main entry point.
- It resolves both commit references, just like other repository methods.
- It finds the common ancestor using the existing `find_common_ancestor_core` function.
- It loads the trees for the ancestor, "ours", and "theirs" commits.
- It calls `merge_trees_core` to build a new merged tree.

Important lines/sections:
- Resolving commit refs: if a ref cannot be resolved, a `RepositoryError` is raised.
- Common ancestor check: if there is no ancestor, merge is refused (this keeps the merge safe and predictable).
- Tree loading: the merge always works on tree objects (not directly on the working directory).

### `merge_trees_core`
This function merges directory trees recursively.
- It collects all file/folder names in the three trees (base/ours/theirs).
- It handles fast-forward cases first:
  - If ours and theirs are the same, keep it.
  - If ours is unchanged from base, take theirs.
  - If theirs is unchanged from base, take ours.
- If both sides are folders, it recurses into those folders.
- If both sides are files (blobs), it calls `merge_blob_text` to do the text merge.
- If there is a type mismatch or a delete/modify conflict, it records a conflict and keeps a deterministic choice.

Important lines/sections:
- The `records_match` helper is used to detect "no change" cases cleanly.
- Recursive merge for trees: this is what makes it a real 3-way merge across folders.
- Conflict tracking: conflicts are stored as file paths so the caller can report them.

### `merge_blob_text`
This function performs the real 3-way merge for file content.
- It reads base/ours/theirs blobs as UTF-8 text.
- It uses `merge3.Merge3` and `merge_lines()` to produce merged output.
- It detects conflicts using `merge_groups()` when available, and falls back to scanning for conflict markers.
- It writes the merged text back into the object store as a new blob.

Important lines/sections:
- `Merge3(base_lines, ours_lines, theirs_lines)`: this is the library call that does the 3-way merge.
- The `conflict` flag decides whether the path is added to the conflicts list.

### `read_blob_text` and `save_blob_text`
These helpers convert between blob hashes and UTF-8 strings.
- `read_blob_text` reads object content and decodes it.
- `save_blob_text` hashes the text and stores it as a new blob.

These are important because the merge operates on text, but the repository stores blobs by hash.

### `hash_string` (plumbing)
This is a small helper around `_libcaf.hash_string`. It allows the merge code to create a new blob hash directly from merged text content.

## How to use (conceptually)
Call `Repository.merge_commits(ref1, ref2)` and you get a `MergeResult` back. The `tree_hash` can be used to create a new commit if needed, and `conflicts` tells you which files need manual attention.
