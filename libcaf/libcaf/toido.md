Hi Ido, we have a question about comment 2. 

We chose to work with merge3 but even if we streamed the file reads from disk instead of loading them all at once, the merge3 algorithm would still need to load all the data into memory to perform its computations. Streaming the I/O would only add complexity without reducing memory usage. thats how merge3 works.

The practical solution would be to add file size limits. For files above a threshold, we could:

Skip the 3-way merge and mark them as conflicts or treat them as binary files but we dont know if thats the purpose.

What do you think? thanks