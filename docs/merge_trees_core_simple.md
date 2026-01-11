# merge_trees_core – Simple Guide

This note explains what `merge_trees_core` does, in very plain words, and why each part exists.

## Big picture
`merge_trees_core` merges **three directory trees**:
- **base** (the common ancestor)
- **ours** (your version)
- **theirs** (the other version)

It returns **one new tree hash** that represents the merged result, and it records any conflicts in a list.

## Inputs (parameters)
- `objects_dir`: folder where all objects (trees/blobs) are stored. It is used to load and save tree/blob data.
- `base_tree`, `ours_tree`, `theirs_tree`: the three trees to merge. Any of them can be `None` (missing tree).
- `path_prefix`: the folder path we are currently inside (used for conflict paths).
- `conflicts`: a list that gets file paths added to it when there is a conflict.

## Output (return value)
- A `HashRef` that points to the **new merged tree**.

## How the function works (step by step)

### 1) Collect all names that appear in any tree
```python
for name in sorted(set(base_records) | set(ours_records) | set(theirs_records)):
```
We want to merge every file/folder that exists in any version.

### 2) Load the three records for this name
```python
base_record = base_records.get(name)
ours_record = ours_records.get(name)
theirs_record = theirs_records.get(name)
```
A record describes either a blob (file) or a tree (folder).

### 3) Fast paths (no real merge needed)
These cases handle "obvious" decisions:

#### a) Ours and theirs are identical
```python
if records_match(ours_record, theirs_record):
```
Keep that exact record.

#### b) Ours did not change from base
```python
if records_match(base_record, ours_record):
```
Take theirs (because ours stayed the same, theirs changed).

#### c) Theirs did not change from base
```python
if records_match(base_record, theirs_record):
```
Take ours.

### 4) Both sides are folders
```python
if ours_record and theirs_record and ... type == TREE:
```
We recursively merge inside that folder:
- Load each subtree
- Call `merge_trees_core` again
- Store the merged subtree hash as a new tree record

### 5) Both sides are files (blobs)
```python
if ours_record and theirs_record and ... type == BLOB:
```
We do a **3‑way text merge** using `merge_blob_text`.
- It returns a new blob hash
- It also tells us if there was a conflict
- If conflict, we add this file path to `conflicts`

### 6) Fallback conflict case
If we reach here, it means:
- One side is a file and the other is a folder, OR
- The file/folder was deleted/added in conflicting ways

We pick one side deterministically (prefers `ours` if it exists), **but** we mark it as a conflict:
```python
chosen = ours_record or theirs_record
conflicts.append(path)
```

### 7) Save the merged tree
At the end we build a new `Tree` and save it:
```python
merged_tree = Tree(merged_records)
save_tree(objects_dir, merged_tree)
return hash_object(merged_tree)
```
This is the hash returned from the function.

## Why conflicts are just paths
The function doesn’t try to resolve every conflict automatically. It only:
- Stores the merged content
- Tells you which files need attention

So the caller can later show conflicts to the user.

## Tiny glossary
- **Tree**: a directory snapshot
- **TreeRecord**: one entry in a tree (file or folder)
- **Blob**: file content stored by hash
- **HashRef**: an object that wraps a hash string

If you want, I can also annotate the exact lines in `libcaf/libcaf/repository.py` next.
