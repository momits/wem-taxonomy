# Towards a Taxonomy of Word Embedding Models: The Database

This repository contains the code that was used to collect relevant 
publications and store them in a Postgres Database.
You are welcome to reproduce our database!
Note however that the search results on Google Scholar may have been updated
since our publication, leading to slightly different results.

As a prerequisite, you must have set up a Postgres database with the proper 
database schema. To do this, you can execute our Postgres schema dump provided 
in `data/wem_taxonomy_schema.sql` with the `psql` shell:
```bash
psql dbname < data/wem_taxonomy_schema.sql
```
where `dbname` is the name of an empty database that you have already created 
for this purpose.
You must also create a user called 'taxonomist' who will own the created
tables.

Next, you should clone this repository.
Then, from within the repository root directory,
pull in the `pubfisher` submodule:
```bash
git submodule update --remote lib/pubfisher
```
Now, install this module from source using pip:
```bash
python3 -m pip install -e lib/pubfisher
```
After that, you can install the `reproduce_wem_taxonomy` package as well:
```bash
python3 -m pip install -e .
```

In order to finally collect the publications,
simply execute the module `fish_wem_taxonomy`:
```bash
python3 -m reproduce_wem_taxonomy.fish_wem_taxonomy
```

The publications are now stored in the database table 'publications' of
your previously created Postgres database.