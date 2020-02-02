# Exported database views

In this directory you find various files exported from our WEM taxonomy database.

## Schema dump

The file `wem_taxonomy_schema.sql` contains only the schema of our database.
You can use this file to replicate the schema of our database and then use
the `reproduce_wem_taxonomy` python package to fill it with up-to-date relevant
WEM publications with the same code that we used for our publication.
Instructions for this are in the top-level `README.md` of this repo. 

## Full database dump

The file `wem_taxonomy_full.sql` is a full dump of our database.
You can use this file to instantiate our database in a Postgres instance
and then explore the database with our [graphical UCC tool][1].

## JSON views 

For convenience, we also provide 3 views on our database as JSON files:

  - `by_publication.json`:
    For each publication that mentions a use case,
    all mentions of use cases in this publication, with the domains where the
    use cases are applied and the models that are used.
    The data schema is as follows:
    ```python
    [
     {  # A publication. 
       
       "pub_id": 518,  # The id of this publication.
                       # Corresponds to the `pub_id` column
                       # of the table `publications`
                       # in the Postgres database.
    
       "abbreviation": "[Faruqui 2014]",  # This publication represented as a
                                          # short "[Author Year]" string.
                                 
       "title": "Retrofitting ...",  # Title of this publication.
       
       "year": 2014,  # Year of this publication.
       
       "authors": "M Faruqui, J Dodge, ...",  # Authors of this publication.
       
       "abstract": "Vector space...",  # Abstract of this publication.
                                       # Formatted as Markdown
                                       # and sometimes annotated
                                       # with additional information.
       
       "url": "https://arxiv.org/...",  # The URL where this publication
                                        # can be accessed.
    
       "citation_count": 545, # The citation count of this publication
                              # on Google Scholar.
    
       "included_because": [  # The reasons why we included this publication into 
                              # our literature collection.
         {
           "retrieval_url": "https://scholar...",  # the URL that was used
                                                   # to retrieve information about
                                                   # this publication
    
           "included_because": "cites",  # A sufficient reason to include
                                         # the publication into our
                                         # literature collection.
                                         # Possible reasons are:
                                         # 
                                         #   "primary":
                                         #     The publication introduces one of
                                         #     the three most important WEMs,
                                         #     namely Word2Vec, GloVE or ElMo.
                                         #   
                                         #   "cites":
                                         #     The publication cites one of the
                                         #     just mentioned three primary 
                                         #     publications.
                                         #   
                                         #   "supplementary":
                                         #     The publication describes a WEM 
                                         #     that is needed to understand 
                                         #     one of the above publications.
    
           "cites_pub_id": 479  # id of the cited publication.
                                # Corresponds to the `pub_id` column
                                # of the table `publications`
                                # in the Postgres database.
                                # This attribute is only provided 
                                # if "included_because" equals "cites".
         }
       ],
       "mentions_use_cases": [  # All mentions of use cases in this publication.
         {
           "mention_id": 72,  # The id of this mention.
                              # Corresponds to the `mention_id` column 
                              # of the table `use_case_mentions`
                              # in the Postgres database.
           
           "mention_description": "Various...",  # Describes how the publication
                                                 # mentions the use case.
                                                 # Formatted as Markdown.
           
           "use_case": {  # A use case mentioned by this publication.
           
             "uc_id": 31,  # The id of this use case.
                           # Corresponds to the `uc_id` column 
                           # of the table `use_cases`
                           # in the Postgres database.
           
             "title": "Interpret...",  # Title of this use case.
             
             "description": "Semantic ..."  # Description of this use case,
                                            # formatted as Markdown.
           },
           "applied_in_domains": [  # The list of domains where the above use case
                                    # was applied according to the above
                                    # publication.
             {
               "domain": {  # A domain.
                 
                 "domain_id": 24,  # The id of this domain.
                                   # Corresponds to the `dom_id` column 
                                   # of the table `domains`
                                   # in the Postgres database.
                 
                 "name": "Semantic Similarity",  # Name of this domain.
                 
                 "description": "Computing ..."  # Description of this domain,
                                                 # formatted as Markdown.
               },
               "application_description": "..."  # Describes how the above use case
                                                 # was applied in the above domain
                                                 # according to the 
                                                 # above publication.
                                                 # Formatted as Markdown.
             }
           ],
           "used_models": [  # The list of models that were used in the
                             # above publication to implement the above use case.
             {
               "model_id": 3,  # The id of this model.
                               # Corresponds to the `model_id` column 
                               # of the table `models`
                               # in the Postgres database.
    
               "name": "Skip-gram",  # The name of this model.
    
               "publication": {  # The publication that introduced this model
                                 # to the research community.
    
                 # ...
                 # The attributes of this publication
                 # are the same as above.
                 # ...
               },
    
               "entity": "Words"  # The entity embedded by this model.
             }
           ]
         }
       ]
     }
    ]
    ```

  - `by_use_case.json`:
    For each use case, all publications mentioning this use case,
    with the domains where they mention the use cases
    and with the specific models they use.
    The data schema is as follows:
    ```python
    [
      {
        "uc_id": 29,  # The id of this use case.
                      # Corresponds to the `uc_id` column
                      # of the table `use_cases`
                      # in the Postgres database.
        
        
        "title": "Learning Embeddings",  # The title of this use case.
        
        "description": "A new model ...",  # Description of this use case,
                                           # formatted as Markdown.
        
        "mentioned_in": [  # All mentions of this use case in our collected publications.
          {
            "mention_id": 35,  # The id of this mention.
                               # Corresponds to the `mention_id` column 
                               # of the table `use_case_mentions`
                               # in the Postgres database.
    
            "mention_description": "Uses skip-gram ...",  # Describes how the
                                                          # below publication
                                                          # mentions this use case.
                                                          # Formatted as Markdown.
            
            "publication": {  # The publication mentioning this use case.
                              # The attributes are the same as in `by_publication.json`.
              "pub_id": 492,
              "abbreviation": "[Frome 2013]",
              "title": "Devise: ...",
              "year": 2013,
              # ... 
            },
            "applied_in_domains": [  # The list of domains where this use case
                                     # was applied according to the above
                                     # publication.
                                     # The attributes are the same 
                                     # as in `by_publication.json`.
              {
                "domain": {
                    # ...
                },
                "application_description": "..."
              }
            ],
            
            "used_models": [  # The list of models that were used in the
                              # above publication to implement the above use case.
                              # The attributes are the same 
                              # as in `by_publication.json`.
            ]
          }
        ]
      }
    ]
    ```

  - `by_domain.json`:
    For each domain, all use cases that are applied in this domain,
    the corresponding publications and the models they use.
    The data schema is as follows:
    ```python
    [
      {  # A domain.
     
        "dom_id": 18,  # The id of this domain.
                       # Corresponds to the `dom_id` column 
                       # of the table `domains`
                       # in the Postgres database.
        
        "name": "Image Captioning & Object Recognition", # Name of this domain.
        
        "description": "Object ...", # Description of this domain,
                                     # formatted as Markdown.
        
        "applied_use_cases": [  # All mentions of use cases that were applied
                                # in this domain in our collected literature.
          {
            "mention_id": 44,  # The id of this mention.
                               # Corresponds to the `mention_id` column 
                               # of the table `use_case_mentions`
                               # in the Postgres database.
            
            "mention_description": "...",  # Describes how the publication
                                           # mentions the use case.
                                           # Formatted as Markdown.
            
            "publication": {  # The publication mentioning the use case
                              # in this domain.
                              # The attributes are the same 
                              # as in `by_publication.json`.
              
              "pub_id": 483,
              "abbreviation": "[Vinyals 2015]",
              "title": "Show and tell: A neural image caption generator",
              # ...
            },
    
            "use_case": { 
              "uc_id": 27,  # The id of this use case.
                            # Corresponds to the `uc_id` column
                            # of the table `use_cases`
                            # in the Postgres database.
              
              "title": "Enhancing the Input",  # The title of this use case.
              
              "description": "..."  # Description of this use case,
                                    # formatted as Markdown.
            },
            
            "used_models": [  # The list of models that were used in the
                              # above publication to implement the above use case.
                              # The attributes are the same 
                              # as in `by_publication.json`.
            ]
          }
        # ...
        ]
      }
    ]
    ```

These views were generated with our [graphical UCC tool][1].
If you install the UCC, you can work on the database and update these views!

[1]: https://git.scc.kit.edu/ukqba/use-case-collector
