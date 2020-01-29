# Exported database views

In this directory you find various files exported from our WEM taxonomy database.

## Schema dump

The file `wem_taxonomy_schema.sql` contains only the schema of our database.
You can use this file to replicate the schema of our database and then use
the `reproduce_wem_taxonomy` python package to fill it with up-to-date relevant
WEM publications with the same code that we used for our publication.
Instructions for this are in the top-level `README.md` of this repo. 

## JSON views 

For convenience, we also provide 3 views on our database as JSON files:

 - `by_publication.json`:
   For each publication that mentions a use case,
   all mentions of use cases in this publication.

 - `by_use_case.json`:
   For each use case, all publications mentioning this use case.

 - `by_domain.json`:
   For each domain, all use cases that are applied in this domain
   and the corresponding publications.

These views were generated with our [graphical UCC tool][1].
If you install the UCC, you can work on the database and update these views!

[1]: https://git.scc.kit.edu/ukqba/use-case-collector
