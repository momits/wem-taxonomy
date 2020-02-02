--
-- PostgreSQL database dump
--

-- Dumped from database version 11.6
-- Dumped by pg_dump version 11.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: embeddings; Type: SCHEMA; Schema: -; Owner: taxonomist
--

CREATE SCHEMA embeddings;


ALTER SCHEMA embeddings OWNER TO taxonomist;

--
-- Name: pg_trgm; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;


--
-- Name: EXTENSION pg_trgm; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pg_trgm IS 'text similarity measurement and index searching based on trigrams';


SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: domain_applications; Type: TABLE; Schema: public; Owner: taxonomist
--

CREATE TABLE public.domain_applications (
    app_use_case_mention integer NOT NULL,
    app_id integer NOT NULL,
    app_domain integer NOT NULL,
    app_description character varying NOT NULL
);


ALTER TABLE public.domain_applications OWNER TO taxonomist;

--
-- Name: domain_applications_app_id_seq; Type: SEQUENCE; Schema: public; Owner: taxonomist
--

CREATE SEQUENCE public.domain_applications_app_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.domain_applications_app_id_seq OWNER TO taxonomist;

--
-- Name: domain_applications_app_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: taxonomist
--

ALTER SEQUENCE public.domain_applications_app_id_seq OWNED BY public.domain_applications.app_id;


--
-- Name: domain_mentions_dmention_id_seq; Type: SEQUENCE; Schema: public; Owner: taxonomist
--

CREATE SEQUENCE public.domain_mentions_dmention_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.domain_mentions_dmention_id_seq OWNER TO taxonomist;

--
-- Name: domain_mentions_dmention_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: taxonomist
--

ALTER SEQUENCE public.domain_mentions_dmention_id_seq OWNED BY public.domain_applications.app_id;


--
-- Name: domains; Type: TABLE; Schema: public; Owner: taxonomist
--

CREATE TABLE public.domains (
    dom_id integer NOT NULL,
    dom_name character varying NOT NULL,
    dom_description character varying NOT NULL,
    dom_short character varying(20),
    dom_super character varying
);


ALTER TABLE public.domains OWNER TO taxonomist;

--
-- Name: domains_dom_id_seq; Type: SEQUENCE; Schema: public; Owner: taxonomist
--

CREATE SEQUENCE public.domains_dom_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.domains_dom_id_seq OWNER TO taxonomist;

--
-- Name: domains_dom_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: taxonomist
--

ALTER SEQUENCE public.domains_dom_id_seq OWNED BY public.domains.dom_id;


--
-- Name: domains_lexeme; Type: VIEW; Schema: public; Owner: taxonomist
--

CREATE VIEW public.domains_lexeme AS
 SELECT ts_stat.word
   FROM ts_stat('SELECT to_tsvector(''simple'', dom_name) ||
                     to_tsvector(''simple'', dom_description)
              FROM domains'::text) ts_stat(word, ndoc, nentry);


ALTER TABLE public.domains_lexeme OWNER TO taxonomist;

--
-- Name: domains_ts_vectors; Type: VIEW; Schema: public; Owner: taxonomist
--

CREATE VIEW public.domains_ts_vectors AS
 SELECT domains.dom_id,
    domains.dom_name,
    domains.dom_description,
    domains.dom_short,
    domains.dom_super,
    to_tsvector('english'::regconfig, (domains.dom_name)::text) AS dom_name_v,
    to_tsvector('english'::regconfig, (domains.dom_description)::text) AS dom_description_v
   FROM public.domains;


ALTER TABLE public.domains_ts_vectors OWNER TO taxonomist;

--
-- Name: models; Type: TABLE; Schema: public; Owner: taxonomist
--

CREATE TABLE public.models (
    model_name character varying NOT NULL,
    model_publication integer NOT NULL,
    model_id integer NOT NULL,
    model_entity character varying DEFAULT ''::character varying NOT NULL
);


ALTER TABLE public.models OWNER TO taxonomist;

--
-- Name: model_types_id_seq; Type: SEQUENCE; Schema: public; Owner: taxonomist
--

CREATE SEQUENCE public.model_types_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.model_types_id_seq OWNER TO taxonomist;

--
-- Name: model_types_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: taxonomist
--

ALTER SEQUENCE public.model_types_id_seq OWNED BY public.models.model_id;


--
-- Name: origins; Type: TABLE; Schema: public; Owner: taxonomist
--

CREATE TABLE public.origins (
    origin_id integer NOT NULL,
    origin_url character varying NOT NULL,
    origin_retrieval_date date NOT NULL,
    origin_cites integer,
    origin_kind character varying DEFAULT 'cites'::character varying NOT NULL,
    origin_comment character varying
);


ALTER TABLE public.origins OWNER TO taxonomist;

--
-- Name: origins_origin_id_seq; Type: SEQUENCE; Schema: public; Owner: taxonomist
--

CREATE SEQUENCE public.origins_origin_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.origins_origin_id_seq OWNER TO taxonomist;

--
-- Name: origins_origin_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: taxonomist
--

ALTER SEQUENCE public.origins_origin_id_seq OWNED BY public.origins.origin_id;


--
-- Name: publications; Type: TABLE; Schema: public; Owner: taxonomist
--

CREATE TABLE public.publications (
    pub_id integer NOT NULL,
    pub_title character varying NOT NULL,
    pub_authors character varying NOT NULL,
    pub_abstract character varying,
    pub_eprint character varying,
    pub_year integer,
    pub_url character varying NOT NULL,
    pub_relevant boolean DEFAULT true NOT NULL,
    pub_citation_count integer
);


ALTER TABLE public.publications OWNER TO taxonomist;

--
-- Name: publication_id_seq; Type: SEQUENCE; Schema: public; Owner: taxonomist
--

CREATE SEQUENCE public.publication_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.publication_id_seq OWNER TO taxonomist;

--
-- Name: publication_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: taxonomist
--

ALTER SEQUENCE public.publication_id_seq OWNED BY public.publications.pub_id;


--
-- Name: publication_origins; Type: TABLE; Schema: public; Owner: taxonomist
--

CREATE TABLE public.publication_origins (
    pub_id integer NOT NULL,
    pub_origin integer NOT NULL,
    pub_origin_position integer NOT NULL
);


ALTER TABLE public.publication_origins OWNER TO taxonomist;

--
-- Name: use_case_mentions; Type: TABLE; Schema: public; Owner: taxonomist
--

CREATE TABLE public.use_case_mentions (
    mention_use_case integer NOT NULL,
    mention_publication integer NOT NULL,
    mention_description character varying DEFAULT ''::character varying,
    mention_id integer NOT NULL
);


ALTER TABLE public.use_case_mentions OWNER TO taxonomist;

--
-- Name: use_cases; Type: TABLE; Schema: public; Owner: taxonomist
--

CREATE TABLE public.use_cases (
    uc_description character varying,
    uc_id integer NOT NULL,
    uc_title character varying NOT NULL,
    uc_short character varying(20)
);


ALTER TABLE public.use_cases OWNER TO taxonomist;

--
-- Name: publications_mention_use_cases; Type: VIEW; Schema: public; Owner: taxonomist
--

CREATE VIEW public.publications_mention_use_cases AS
 SELECT p.pub_id,
    p.pub_title,
    p.pub_authors,
    p.pub_abstract,
    p.pub_eprint,
    p.pub_year,
    p.pub_url,
    p.pub_relevant,
    p.pub_citation_count,
    m.mention_use_case,
    m.mention_publication,
    m.mention_description,
    m.mention_id,
    u.uc_description,
    u.uc_id,
    u.uc_title,
    u.uc_short
   FROM public.publications p,
    public.use_case_mentions m,
    public.use_cases u
  WHERE ((p.pub_id = m.mention_publication) AND (u.uc_id = m.mention_use_case));


ALTER TABLE public.publications_mention_use_cases OWNER TO taxonomist;

--
-- Name: publications_missing_origins; Type: VIEW; Schema: public; Owner: taxonomist
--

CREATE VIEW public.publications_missing_origins AS
 SELECT pub.pub_id,
    pub.pub_title,
    pub.pub_authors,
    pub.pub_abstract,
    pub.pub_eprint,
    pub.pub_year,
    pub.pub_url,
    pub.pub_relevant,
    pub.pub_citation_count,
    po.pub_origin_position,
    o.origin_id,
    o.origin_url,
    o.origin_retrieval_date,
    o.origin_cites,
    o.origin_kind
   FROM ((public.publications pub
     LEFT JOIN public.publication_origins po ON ((pub.pub_id = po.pub_id)))
     LEFT JOIN public.origins o ON ((po.pub_origin = o.origin_id)))
  WHERE (po.pub_origin IS NULL);


ALTER TABLE public.publications_missing_origins OWNER TO taxonomist;

--
-- Name: publications_with_origins; Type: VIEW; Schema: public; Owner: taxonomist
--

CREATE VIEW public.publications_with_origins AS
 SELECT pub.pub_id,
    pub.pub_title,
    pub.pub_authors,
    pub.pub_abstract,
    pub.pub_eprint,
    pub.pub_year,
    pub.pub_url,
    pub.pub_relevant,
    pub.pub_citation_count,
    po.pub_origin_position,
    o.origin_id,
    o.origin_url,
    o.origin_retrieval_date,
    o.origin_cites,
    o.origin_kind,
    o.origin_comment
   FROM ((public.publications pub
     JOIN public.publication_origins po ON ((pub.pub_id = po.pub_id)))
     JOIN public.origins o ON ((po.pub_origin = o.origin_id)));


ALTER TABLE public.publications_with_origins OWNER TO taxonomist;

--
-- Name: use_case_mention_mention_id_seq; Type: SEQUENCE; Schema: public; Owner: taxonomist
--

CREATE SEQUENCE public.use_case_mention_mention_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.use_case_mention_mention_id_seq OWNER TO taxonomist;

--
-- Name: use_case_mention_mention_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: taxonomist
--

ALTER SEQUENCE public.use_case_mention_mention_id_seq OWNED BY public.use_case_mentions.mention_id;


--
-- Name: use_case_mention_models; Type: TABLE; Schema: public; Owner: taxonomist
--

CREATE TABLE public.use_case_mention_models (
    mention_id integer NOT NULL,
    mention_model integer NOT NULL
);


ALTER TABLE public.use_case_mention_models OWNER TO taxonomist;

--
-- Name: use_case_mentions_use_models; Type: VIEW; Schema: public; Owner: taxonomist
--

CREATE VIEW public.use_case_mentions_use_models AS
 SELECT use_case_mentions.mention_id,
    use_case_mentions.mention_use_case,
    use_case_mentions.mention_publication,
    use_case_mentions.mention_description,
    models.model_id,
    models.model_name,
    models.model_publication,
    models.model_entity
   FROM public.use_case_mentions,
    public.use_case_mention_models,
    public.models
  WHERE ((use_case_mentions.mention_id = use_case_mention_models.mention_id) AND (models.model_id = use_case_mention_models.mention_model));


ALTER TABLE public.use_case_mentions_use_models OWNER TO taxonomist;

--
-- Name: use_cases_applied_in_domains; Type: VIEW; Schema: public; Owner: taxonomist
--

CREATE VIEW public.use_cases_applied_in_domains AS
 SELECT domain_applications.app_id,
    domain_applications.app_description,
    domains.dom_id,
    domains.dom_short,
    domains.dom_name,
    domains.dom_super,
    domains.dom_description,
    use_case_mentions.mention_id,
    use_case_mentions.mention_use_case,
    use_case_mentions.mention_publication,
    use_case_mentions.mention_description,
    use_cases.uc_id,
    use_cases.uc_short,
    use_cases.uc_title,
    use_cases.uc_description,
    publications.pub_id,
    publications.pub_title,
    publications.pub_year,
    publications.pub_authors,
    publications.pub_citation_count,
    publications.pub_relevant
   FROM ((((public.domain_applications
     JOIN public.domains ON ((domain_applications.app_domain = domains.dom_id)))
     JOIN public.use_case_mentions ON ((domain_applications.app_use_case_mention = use_case_mentions.mention_id)))
     JOIN public.use_cases ON ((use_case_mentions.mention_use_case = use_cases.uc_id)))
     JOIN public.publications ON ((use_case_mentions.mention_publication = publications.pub_id)));


ALTER TABLE public.use_cases_applied_in_domains OWNER TO taxonomist;

--
-- Name: use_cases_lexeme; Type: VIEW; Schema: public; Owner: taxonomist
--

CREATE VIEW public.use_cases_lexeme AS
 SELECT ts_stat.word
   FROM ts_stat('SELECT to_tsvector(''simple'', uc_title) ||
                     to_tsvector(''simple'', uc_description)
              FROM use_cases'::text) ts_stat(word, ndoc, nentry);


ALTER TABLE public.use_cases_lexeme OWNER TO taxonomist;

--
-- Name: use_cases_ts_vectors; Type: VIEW; Schema: public; Owner: taxonomist
--

CREATE VIEW public.use_cases_ts_vectors AS
 SELECT use_cases.uc_description,
    use_cases.uc_id,
    use_cases.uc_title,
    use_cases.uc_short,
    to_tsvector('english'::regconfig, (use_cases.uc_title)::text) AS uc_title_v,
    to_tsvector('english'::regconfig, (use_cases.uc_description)::text) AS uc_description_v
   FROM public.use_cases;


ALTER TABLE public.use_cases_ts_vectors OWNER TO taxonomist;

--
-- Name: use_cases_uc_id_seq; Type: SEQUENCE; Schema: public; Owner: taxonomist
--

CREATE SEQUENCE public.use_cases_uc_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.use_cases_uc_id_seq OWNER TO taxonomist;

--
-- Name: use_cases_uc_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: taxonomist
--

ALTER SEQUENCE public.use_cases_uc_id_seq OWNED BY public.use_cases.uc_id;


--
-- Name: domain_applications app_id; Type: DEFAULT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.domain_applications ALTER COLUMN app_id SET DEFAULT nextval('public.domain_applications_app_id_seq'::regclass);


--
-- Name: domains dom_id; Type: DEFAULT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.domains ALTER COLUMN dom_id SET DEFAULT nextval('public.domains_dom_id_seq'::regclass);


--
-- Name: models model_id; Type: DEFAULT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.models ALTER COLUMN model_id SET DEFAULT nextval('public.model_types_id_seq'::regclass);


--
-- Name: origins origin_id; Type: DEFAULT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.origins ALTER COLUMN origin_id SET DEFAULT nextval('public.origins_origin_id_seq'::regclass);


--
-- Name: publications pub_id; Type: DEFAULT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.publications ALTER COLUMN pub_id SET DEFAULT nextval('public.publication_id_seq'::regclass);


--
-- Name: use_case_mentions mention_id; Type: DEFAULT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.use_case_mentions ALTER COLUMN mention_id SET DEFAULT nextval('public.use_case_mention_mention_id_seq'::regclass);


--
-- Name: use_cases uc_id; Type: DEFAULT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.use_cases ALTER COLUMN uc_id SET DEFAULT nextval('public.use_cases_uc_id_seq'::regclass);


--
-- Data for Name: domain_applications; Type: TABLE DATA; Schema: public; Owner: taxonomist
--

COPY public.domain_applications (app_use_case_mention, app_id, app_domain, app_description) FROM stdin;
37	2	6	SGNS are trained for each language. Then a linear mapping between the two vector spaces is learned (optimizing the vector distances to the correct translations on the training data).
49	19	15	Pre-trained GloVe embeddings are used to represent words, which are then fed into a neural network to obtain sentence embeddings.\nThe sentence embeddings are then used to *train a neural network on the new SNLI textual entailment task, which for the first time reaches performance close to a lexicalized model.*
62	31	16	Improvements in SQuaD (2016) and adversarial SQuaD (2017).
53	33	23	Embeddings of nodes are initialized using GloVe.
50	20	12	Achieves SOTA performance on CoNLL 2003 corpus: 91.21% F1
41	32	18	Given an existing image classifier producing a distribution over all possible captions, the **embeddings corresponding to the caption words are weighted according to the captions probability**, then summed up to form the predicted embedding.\nThen, a **k-NN search is used to determine the k most probable classes**.\n\nReaches a larger *hierarchical precision* on unseen images than *DeViSe* and even outperforms the original model on the original classes (w.r.t hierarchical precision).\n\nA nice example of *zero-shot learning*:\nthe interpolation between word vectors is able to predict captions which never occurred in the training data.
35	15	18	Achieves state-of-the-art performance on the 1000-class ImageNet object recognition challenge (ILSVRC 2012 1K).
57	27	18	Class labels are embedded (e.g. with word2vec), then a model is trained to learn the *compatibility* between image embeddings and class embeddings.
44	16	18	Full sentences describing images are produced.
50	21	20	Achieves SOTA performance on Penn Treebank WSJ corpus: 97.55% accuracy
58	28	15	ELMO increases performance on MNLI and QNLI.
54	23	21	Query predicates are encoded as vectors using the embeddings of the values contained in the predicate.
36	7	12	SGNS help to improve the performance.
47	17	18	Full sentences to describe images are generated.\nObtains state of the art results on Flickr8K, Flickr30K and MSCOCO.\n
52	22	18	Outperforms SOTA systems on Flickr30k and COCO.
46	10	13	Re-implementation of *He et al. (2017)*\n*OntoNotes benchmark (Pradhan et al., 2012)*\n\n8-layer deep biLSTM with forward and backward directions interleaved\n\nELMO achieves 1.2% relative increase of F1.
46	11	14	*End-to-end span-based neural model of Lee et al. (2017):*\n*OntoNotes coreference task (Pradhan et al., 2012)*\n\nUses biLSTM and attention mechanism to first compute span representations, then applies a softmax mention ranking model to find coreference chains.\n\nELMO achieves 1.6% relative increase of F1.
46	12	15	*Stanford Natural Language Inference (SNLI) corpus (Bowman et al., 2015):*\n\nProvides approximately 550K hypothesis/premise pairs.\n\nELMO achieves 0.7%% relative increase of accuracy.
46	13	16	*SQuAD dataset:*\n\n100K+ crowd sourced question-answer pairs where the answer is a span in a given Wikipedia paragraph.\n\n25% error reduction attributable to ELMO.
61	30	13	ELMO improves the performance on CoNLL-2005 and CoNLL-2012.
39	36	24	Use skip-gram embeddings to derive a similarity metric between documents which they call *Word Mover's Distance* (WMD).\n\nCompare WMD with other metrics by measuring the\nperformance of various k-NN classification problems of documents.\n**WMD outperforms all 7 state-of-the-art alternative document distances in 6 of 8 real world classification tasks**.\nThus, the distance vectors produced by the skip-gram embeddings can be used to effectively classify documents.
63	34	23	Embeddings of neighboring nodes are used to predict an embedding of the target node.
45	37	24	Annotate images with dense descriptions of the entities visible in specific regions and their semantic relationships.\n\nTo assess the quality of their annotations, they want to analyse the "semantic diversity" of the region descriptions, like this:\n\n 1. Map each word of a region description to the corresponding\n     skip-gram embedding (using the word2vec pre-trained model).\n 2. Average of all embeddings = region embedding\n 3. Cluster the region embeddings with hierarchical agglomerative\n     clustering (assumption: distances between embeddings\n     represent semantic dissimilarity).\n\nObtain 71 "semantic/syntactic clusters".\nDiversity: on average, each image contains descriptions from 17 different clusters.
38	38	6	A simple linear mapping between the vector spaces of the WEMs of two different languages is learned and used for translation between those languages.
60	39	16	MQAN
60	40	12	MQAN
60	41	15	MQAN
60	44	6	MQAN
42	45	24	Embed words in the same vector space as visuals.\nAccording to the authors, computing\n```\n  e(image of a blue car) - e(the word "blue") + e(the word "red")\n```\nproduces an embedding close to `e(image of a red car)`.
64	46	24	According to the authors, computing\n```\n  e(king) - e(queen) + e(actor)\n```\nproduces an embedding close to `e(actress)`.
56	47	25	
65	48	26	Sentence parse-tree is linearized and fed into a neural network. The words are embedded with skip-gram.
66	49	27	Embeddings are used to improve CNN for sentence classification tasks.
67	50	12	Embeddings enhance the input for NER.
43	3	27	PEM reduces error rate on SST-5 and improves performance on IMDB data.
58	29	27	ELMO increases performance on SST-2.
75	62	26	Embeddings serve as inputs to a multi-lingual dependency parser.
76	63	26	Embeddings help to implement a dependency parser that can parse multiple languages.
60	42	27	MQAN
48	18	27	Stanford Sentiment Treebank:\n*Outperforms SOTA systems on fine-grained classification (SST-5)*
81	68	28	Nearest neighbouring materials to specific wanted properties are searched in the embedding space.
85	71	12	GloVe embeddings increase performance of CRF model in NER.
86	72	24	Distances between words are interpreted as their semantic dissimilarity.
87	73	15	Embeddings improve performance on paraphrase detection (SICK dataset).
88	75	27	Skip-thought embeddings of sentences are used in various classification tasks.
79	66	24	Bilingual word vectors perform better in similarity benchmarks:\n\n - correlation on various word similarity datasets\n - Mikolov's analogy tests
46	9	27	*Sentiment classification task in the Stanford Sentiment Treebank (SST-5; Socher et al., 2013):*\n\nSelecting one of five labels (from very negative to very\npositive) to describe a sentence from a movie re-\nview.\n\nELMO achieves 1% *absolute* increase of accuracy.
67	52	6	Embeddings can be used to translate phrases (maximize cosine distance between phrase embeddings, which are simply the averaged word embeddings).
74	61	24	Authors define a stronger version of semantic similarity than could be found in the previous benchmarks\n -> show that existing models are not so good in capturing similarity\n     (existing models tend to capture relatedness instead).
34	8	27	"initializing word-embeddings using unsupervised pre-training gives an *absolute accuracy increase of around 4.5* when compared to randomly initializing the word-embeddings"\n(on the SST-5 benchmark)
70	56	24	Embeddings are evaluated on two analogy-based word similarity tasks, one from Microsoft and one from Google. k-NN is used on cosine distance to find the predicted word.\n-> assumption: cosine distance = semantic distance
72	58	24	Run a variety of different benchmarks to measure word similarity:\n\n - WS-353 dataset (Finkelstein etal., 2001) \n - RG-65 (Rubensteinand Goodenough, 1965) dataset \n - MEN dataset (Bruni et al., 2012)\n - Mikolovs analogy tests\n - synonym selection (TOEFL)  (Landauer and Dumais, 1997)\n\nRetrofitted embeddings improve performance on these benchmarks.
88	74	18	The sentence describing an image best out of a large set of sentences is chosen using its skip-thought representation.\nI.e. for each sentence embedding, a compatibility score with the image embedding is computed, then the max is chosen.
77	64	6	A linear mapping is learned to translate between embeddings of each language.
91	80	25	Model using skip-grams performs well on COCO-QA and DAQUAR benchmarks.
80	67	24	Sentence embeddings perform well on a benchmark using 121 languages.\nFor each language, up to 1000 pairs of an English sentences the corresponding similiar sentence from the other language are available.\nEvaluation is done by finding the nearest neighbor for each English sentence in the other language (and vice versa) according to cosine similarity and computing the error rate.
68	53	27	Sentence embeddings are used to classify sentences:\n\nA) Predict age of writer of sentence.\nB) Predict sentiment of sentence.
83	69	24	Distances are interpreted as semantic similarity.
93	82	24	 New way of computing the distance between embeddings yields improvements on MSR and GOOGLE datasets.
78	65	6	Mapping outperforms other linear mappings for several translation tasks.
73	60	27	Embeddings improve scores for sentiment analysis (model: l2-regularized logistic regression), on treebank by Socher et al. (2013).
71	57	24	Cosine distance is computed between embeddings of a pair of English-Chinese words, which are known to be translations of each other. This is reported as Alignment Error Rate.\nBilingual embedings reduce this AER.
68	54	15	Textual entailment with SNLI: Predict for a pair of sentences their logical relationship.
69	55	20	Embeddings are input to NN for POS tagging.
84	70	24	GloVe embeddings are used in Mikolovs analogy task.\nAlso some word similarity benchmarks are computed, e.g. WordSim-353.
89	76	16	Ability to answer questions is benchmarked on the acebook bAbI dataset.
89	77	27	Stanford Sentiment Treebank, sentiment classification.
89	78	20	WSJ-PTB benchmark.
92	81	24	Measure the semantic similarity between the query and candidate documents.\nHowever, the dataset the authors used for their evaluation is unclear.
90	79	20	WSJ-PTB benchmark.
94	83	29	Skip-grams improve performance of the model on an evaluation using Freebase relations with the NYT corpus.
\.


--
-- Data for Name: domains; Type: TABLE DATA; Schema: public; Owner: taxonomist
--

COPY public.domains (dom_id, dom_name, dom_description, dom_short, dom_super) FROM stdin;
18	Image Captioning & Object Recognition	Object Recognition:\nDetecting instances of semantic objects (and their relationships) in digital images and videos.\n\nImage Captioning:\nGenerating a natural language description of an image.	ImgCaption	Vision
21	Query Optimization	Computing the fastest execution plan for a database query.	QueryOpt	Databases
25	Visual Question Answering	Highlighting the part of an image that contains the answer to a natural language question (if any) and generating a natural language answer.	VisualQAns	Vision
16	Question Answering	Highlighting the part of a document that contains the answer to a given question (if any).	QAns	NLP
23	Node Labeling	Assigning a label to a node in a graph using information from neighboring nodes.	NodeLabel	Graphs
24	Semantic Similarity	Computing the distance in meaning between two entities, e.g. between two pieces of text or between an image and a text.	Similarity	NLP
26	Parsing	Extracting a parse tree from a sentence that represents its syntactic structure.\nThe resulting parse tree can be constituency based or dependency based.	Parsing	NLP
27	Text Classification	Assigning a class out of a set of classes to a piece of text. E.g. assigning one of multiple sentiments.	TextClassif	NLP
13	Semantic Role Labeling	Extracting triples (subject, predicate, object) from text.	SemRoleLabel	NLP
28	Material Prediction	Predicting materials that are likely to be useful in the future.	MatPredict	Materials
14	Coreference Resolution	Marking multiple occurrences of the same entities in text with the same identifier.	Coreference	NLP
29	Relationship Extraction	Extracting semantic relationships from a text, e.g. married to, lives in.	RelExtract	NLP
20	Part-of-speech Tagging	Assigning a part-of-speech (POS) to each word in a text (POS = a category of words which have similar grammatical properties).	POSTagging	NLP
15	Natural Language Inference	Determining the logical relationship between text snippets, e.g. entailment, contradiction or independence.	NLInference	NLP
12	Named-Entity Recognition	Determining which items in a text map to proper names, such as people or places, and what the type of each such name is (e.g. person, location, organization).	NamedEntity	NLP
6	Translation	Mapping text of one language to text of another language, such that both have similar meanings.	Translation	NLP
\.


--
-- Data for Name: models; Type: TABLE DATA; Schema: public; Owner: taxonomist
--

COPY public.models (model_name, model_publication, model_id, model_entity) FROM stdin;
Paragraph Embeddings	482	12	Paragraphs
ELMo	1794	1	Words
GloVe	479	6	Words
Bilingual Robust Projection	748	14	Words
Character-based	524	10	Words
Bilingually trained	519	11	Words
Multilingual	2480	18	Sentences
Character-based	2580	19	Words
Skip-Thought	491	20	Sentences
Bilingual based on CCA	2452	17	Words
Multimodal	506	5	Words + Images
2-dimensional	531	9	Sentences
Multimodal	2044	16	Words + Images
Retrofitted using Semantic Lexicon	518	13	Words
Subword-based	489	21	Words
Skip-gram	1805	3	Words
CBOW	1805	23	Words
LSTM-RNN	529	24	Sentences
\.


--
-- Data for Name: origins; Type: TABLE DATA; Schema: public; Owner: taxonomist
--

COPY public.origins (origin_id, origin_url, origin_retrieval_date, origin_cites, origin_kind, origin_comment) FROM stdin;
51	https://scholar.google.com/scholar?q=Glove%3A%20Global%20vectors%20for%20word%20representation&hl=en	2020-01-08	\N	primary	GloVe paper.
52	https://scholar.google.com/scholar?q=Deep%20contextualized%20word%20representations&hl=en	2020-01-08	\N	primary	ELMO paper.
55	https://scholar.google.com/scholar?&cites=14181983828043963745&hl=en	2020-01-08	1794	cites	Citations of ELMO paper.
58	https://scholar.google.com/scholar?q=A%20representation%20learning%20framework%20for%20multi-source%20transfer%20parsing&hl=en	2020-01-08	1805	supplementary	Defines a WEM referenced in other publications.
54	https://scholar.google.com/scholar?&cites=15824805022753088965&hl=en	2020-01-08	479	cites	Citations of GloVe paper.
59	https://scholar.google.com/scholar?q=Finding%20Function%20in%20Form%3A%20Compositional%20Character%20Models%20for%20Open%20Vocabulary%20Word%20Representation&hl=en	2020-01-08	1805	supplementary	Defines a WEM referenced in other publications.
57	https://scholar.google.com/scholar?q=Improving%20Vector%20Space%20Word%20Representations%20Using%20Multilingual%20Correlation&hl=en	2020-01-08	1805	cites	Citation of word2vec paper missed out by GS.
56	https://scholar.google.com/scholar?q=Advances%20in%20pre-training%20distributed%20word%20representations&hl=en	2020-01-08	1805	cites	Citation of word2vec paper missed out by GS.
50	https://scholar.google.com/scholar?q=Efficient%20estimation%20of%20word%20representations%20in%20vector%20space&hl=en	2020-01-08	\N	primary	word2vec paper.
53	https://scholar.google.com/scholar?&cites=7447715766504981253&hl=en	2020-01-08	1805	cites	Citations of word2vec paper.
\.


--
-- Data for Name: publication_origins; Type: TABLE DATA; Schema: public; Owner: taxonomist
--

COPY public.publication_origins (pub_id, pub_origin, pub_origin_position) FROM stdin;
1805	50	1
479	51	1
1794	52	1
476	53	1
477	53	2
478	53	3
479	53	4
481	53	5
480	53	6
482	53	7
2472	53	8
484	53	9
483	53	10
485	53	11
487	53	12
489	53	13
488	53	14
486	53	15
490	53	16
494	53	17
496	53	18
491	53	19
2473	53	20
492	53	21
493	53	22
495	53	23
497	53	24
503	53	25
507	53	26
501	53	27
499	53	28
505	53	29
504	53	30
512	53	31
502	53	32
500	53	33
517	53	34
508	53	35
506	53	36
514	53	37
522	53	38
509	53	39
510	53	40
515	53	41
513	53	42
511	53	43
531	53	44
526	53	45
533	53	46
558	53	47
516	53	48
520	53	49
518	53	50
555	53	51
521	53	52
523	53	53
538	53	54
530	53	55
519	53	56
524	53	57
529	53	58
527	53	59
525	53	60
551	53	61
545	53	62
536	53	63
528	53	64
534	53	65
547	53	66
553	53	67
599	53	68
532	53	69
557	53	70
591	53	71
544	53	72
549	53	73
550	53	74
556	53	75
562	53	76
541	53	77
535	53	78
577	53	79
540	53	80
614	53	81
568	53	82
542	53	83
554	53	84
584	53	85
664	53	86
537	53	87
578	53	88
624	53	89
565	53	90
559	53	91
539	53	92
543	53	93
587	53	94
582	53	95
561	53	96
622	53	97
560	53	98
567	53	99
623	53	100
576	53	101
616	53	102
546	53	103
548	53	104
566	53	105
592	53	106
564	53	107
581	53	108
632	53	109
571	53	110
638	53	111
563	53	112
602	53	113
575	53	114
683	53	115
573	53	116
678	53	117
786	53	118
2474	53	119
579	53	120
595	53	121
596	53	122
600	53	123
650	53	124
569	53	125
585	53	126
629	53	127
601	53	128
708	53	129
552	53	130
607	53	131
583	53	132
572	53	133
605	53	134
680	53	135
613	53	136
588	53	137
580	53	138
574	53	139
586	53	140
594	53	141
593	53	142
620	53	143
612	53	144
697	53	145
609	53	146
590	53	147
604	53	148
630	53	149
675	53	150
687	53	151
570	53	152
625	53	153
606	53	154
652	53	155
603	53	156
639	53	157
651	53	158
655	53	159
701	53	160
2581	53	161
634	53	162
667	53	163
621	53	164
660	53	165
631	53	166
610	53	167
597	53	168
644	53	169
659	53	170
643	53	171
755	53	172
633	53	173
628	53	174
691	53	175
598	53	176
2475	53	177
626	53	178
646	53	179
656	53	180
617	53	181
647	53	182
706	53	183
688	53	184
676	53	185
627	53	186
618	53	187
640	53	188
641	53	189
619	53	190
608	53	191
2582	53	192
636	53	193
707	53	194
671	53	195
611	53	196
653	53	197
702	53	198
723	53	199
672	53	200
615	53	201
657	53	202
757	53	203
690	53	204
637	53	205
666	53	206
669	53	207
698	53	208
662	53	209
642	53	210
658	53	211
670	53	212
654	53	213
673	53	214
668	53	215
674	53	216
2583	53	217
2584	53	218
677	53	219
635	53	220
661	53	221
741	53	222
731	53	223
684	53	224
694	53	225
2585	53	226
877	53	227
930	53	228
648	53	229
716	53	230
722	53	231
2586	53	232
645	53	233
685	53	234
686	53	235
2587	53	236
681	53	237
709	53	238
2588	53	239
2589	53	240
649	53	241
703	53	242
730	53	243
736	53	244
665	53	245
682	53	246
705	53	247
696	53	248
713	53	249
2590	53	250
693	53	251
663	53	252
679	53	253
695	53	254
768	53	255
2476	53	256
700	53	257
2591	53	258
777	53	259
2592	53	260
2593	53	261
2594	53	262
2595	53	263
2596	53	264
704	53	265
2597	53	266
2598	53	267
2599	53	268
2600	53	269
2601	53	270
699	53	271
689	53	272
726	53	273
2602	53	274
710	53	275
725	53	276
2603	53	277
880	53	278
715	53	279
2604	53	280
2605	53	281
2606	53	282
2607	53	283
748	53	284
2608	53	285
2609	53	286
728	53	287
770	53	288
742	53	289
2610	53	290
2611	53	291
2477	53	292
729	53	293
2612	53	294
2613	53	295
2614	53	296
2615	53	297
2616	53	298
2617	53	299
2618	53	300
2619	53	301
2620	53	302
2621	53	303
2622	53	304
692	53	305
718	53	306
2623	53	307
745	53	308
2624	53	309
2625	53	310
2626	53	311
2627	53	312
717	53	313
734	53	314
793	53	315
737	53	316
2628	53	317
2629	53	318
2630	53	319
2631	53	320
724	53	321
2632	53	322
2633	53	323
2634	53	324
2635	53	325
2636	53	326
782	53	327
803	53	328
2637	53	329
2638	53	330
2639	53	331
2640	53	332
2641	53	333
2642	53	334
2643	53	335
759	53	336
2644	53	337
792	53	338
2645	53	339
2646	53	340
2647	53	341
784	53	342
2648	53	343
2649	53	344
2650	53	345
2651	53	346
2652	53	347
2653	53	348
837	53	349
2654	53	350
2655	53	351
2656	53	352
2657	53	353
824	53	354
2658	53	355
2659	53	356
2660	53	357
2661	53	358
2662	53	359
2663	53	360
2664	53	361
2665	53	362
2666	53	363
2667	53	364
2668	53	365
2669	53	366
2670	53	367
2671	53	368
2672	53	369
2673	53	370
859	53	371
2674	53	372
2675	53	373
2676	53	374
2677	53	375
2678	53	376
2679	53	377
2680	53	378
2681	53	379
2682	53	380
2683	53	381
834	53	382
818	53	383
2684	53	384
2685	53	385
2686	53	386
2687	53	387
2688	53	388
2689	53	389
820	53	390
2690	53	391
2691	53	392
2692	53	393
2693	53	394
2694	53	395
2695	53	396
2696	53	397
2697	53	398
2698	53	399
2699	53	400
2044	54	1
488	54	2
1794	54	3
2046	54	4
2045	54	5
2051	54	6
2049	54	7
2047	54	8
499	54	9
512	54	10
517	54	11
2048	54	12
2059	54	13
509	54	14
2050	54	15
2052	54	16
531	54	17
2056	54	18
2053	54	19
526	54	20
533	54	21
558	54	22
520	54	23
2055	54	24
518	54	25
1263	54	26
2054	54	27
2060	54	28
2057	54	29
2069	54	30
2063	54	31
2058	54	32
2065	54	33
2068	54	34
2061	54	35
2072	54	36
2062	54	37
2064	54	38
545	54	39
2073	54	40
2067	54	41
2070	54	42
2066	54	43
2079	54	44
2085	54	45
591	54	46
2071	54	47
541	54	48
2076	54	49
577	54	50
614	54	51
2075	54	52
2074	54	53
2077	54	54
2078	54	55
2084	54	56
2080	54	57
2083	54	58
624	54	59
2089	54	60
2081	54	61
2088	54	62
561	54	63
1267	54	64
2110	54	65
2082	54	66
2087	54	67
2099	54	68
571	54	69
2103	54	70
2106	54	71
2095	54	72
2104	54	73
2086	54	74
2098	54	75
2112	54	76
786	54	77
2474	54	78
2091	54	79
595	54	80
2094	54	81
2090	54	82
600	54	83
650	54	84
2093	54	85
2116	54	86
629	54	87
708	54	88
2092	54	89
572	54	90
2096	54	91
680	54	92
588	54	93
580	54	94
2246	54	95
593	54	96
2101	54	97
604	54	98
687	54	99
2124	54	100
2100	54	101
606	54	102
603	54	103
2102	54	104
2113	54	105
2105	54	106
2097	54	107
2123	54	108
655	54	109
2120	54	110
2131	54	111
2117	54	112
2149	54	113
2109	54	114
2155	54	115
631	54	116
2111	54	117
659	54	118
2130	54	119
633	54	120
628	54	121
656	54	122
647	54	123
706	54	124
2133	54	125
2115	54	126
2107	54	127
640	54	128
2164	54	129
636	54	130
2118	54	131
2121	54	132
2127	54	133
2146	54	134
723	54	135
2108	54	136
2143	54	137
2136	54	138
2137	54	139
2135	54	140
2144	54	141
2125	54	142
2126	54	143
669	54	144
698	54	145
2153	54	146
2174	54	147
662	54	148
2129	54	149
642	54	150
670	54	151
2114	54	152
2138	54	153
2119	54	154
677	54	155
2142	54	156
2154	54	157
2141	54	158
2122	54	159
877	54	160
930	54	161
716	54	162
722	54	163
2175	54	164
2157	54	165
2158	54	166
2151	54	167
2128	54	168
2170	54	169
736	54	170
2139	54	171
2171	54	172
2166	54	173
2140	54	174
2145	54	175
2147	54	176
768	54	177
700	54	178
2179	54	179
777	54	180
2132	54	181
2148	54	182
2163	54	183
2185	54	184
2134	54	185
726	54	186
2172	54	187
2160	54	188
880	54	189
715	54	190
2150	54	191
2152	54	192
728	54	193
770	54	194
2173	54	195
2191	54	196
742	54	197
2168	54	198
729	54	199
2188	54	200
2156	54	201
2183	54	202
718	54	203
2180	54	204
2197	54	205
745	54	206
2162	54	207
2159	54	208
2193	54	209
717	54	210
734	54	211
793	54	212
2184	54	213
2195	54	214
737	54	215
2178	54	216
2199	54	217
2478	54	218
2165	54	219
2176	54	220
724	54	221
2177	54	222
2181	54	223
2182	54	224
782	54	225
803	54	226
2169	54	227
2161	54	228
2167	54	229
759	54	230
792	54	231
784	54	232
2186	54	233
837	54	234
2187	54	235
2189	54	236
2190	54	237
2192	54	238
824	54	239
859	54	240
2196	54	241
834	54	242
818	54	243
2198	54	244
2194	54	245
2479	54	246
820	54	247
2480	54	248
2481	54	249
2200	54	250
1263	55	1
1267	55	2
1264	55	3
1279	55	4
1411	55	5
1265	55	6
1269	55	7
2482	55	8
1273	55	9
1202	55	10
1284	55	11
1282	55	12
1266	55	13
1271	55	14
1270	55	15
1307	55	16
2480	55	17
1348	55	18
1272	55	19
1294	55	20
1300	55	21
2483	55	22
1278	55	23
1280	55	24
1283	55	25
1309	55	26
1277	55	27
2484	55	28
1290	55	29
1276	55	30
1295	55	31
1398	55	32
1408	55	33
1292	55	34
1496	55	35
2485	55	36
2486	55	37
1289	55	38
1275	55	39
1293	55	40
1312	55	41
1385	55	42
1304	55	43
1306	55	44
1322	55	45
1308	55	46
1344	55	47
1706	55	48
1297	55	49
1514	55	50
1770	55	51
1286	55	52
1468	55	53
2487	55	54
1369	55	55
1527	55	56
1376	55	57
1274	55	58
1281	55	59
1311	55	60
2489	55	61
2488	55	62
1305	55	63
1354	55	64
2490	55	65
1303	55	66
2491	55	67
1747	55	68
1288	55	69
1316	55	70
1285	55	71
1330	55	72
2492	55	73
2493	55	74
1298	55	75
1299	55	76
1321	55	77
2494	55	78
1342	55	79
2495	55	80
2496	55	81
1350	55	82
2497	55	83
1318	55	84
1310	55	85
1397	55	86
2498	55	87
2499	55	88
1301	55	89
1413	55	90
1337	55	91
2500	55	92
1291	55	93
1287	55	94
1371	55	95
1360	55	96
1326	55	97
1768	55	98
2503	55	99
2501	55	100
1296	55	101
1393	55	102
1317	55	103
1412	55	104
1403	55	105
1707	55	106
1339	55	107
1463	55	108
2502	55	109
2504	55	110
1730	55	111
1380	55	112
1340	55	113
1483	55	114
1329	55	115
2505	55	116
2506	55	117
1315	55	118
1375	55	119
1313	55	120
1624	55	121
2507	55	122
1343	55	123
1325	55	124
1723	55	125
1363	55	126
1365	55	127
2508	55	128
1331	55	129
2509	55	130
1345	55	131
2510	55	132
1328	55	133
1610	55	134
1392	55	135
1409	55	136
1373	55	137
2511	55	138
1404	55	139
2512	55	140
2513	55	141
1323	55	142
1314	55	143
1352	55	144
1429	55	145
2516	55	146
2514	55	147
1689	55	148
1320	55	149
1357	55	150
1359	55	151
2521	55	152
2515	55	153
2517	55	154
1335	55	155
2518	55	156
1379	55	157
1338	55	158
1638	55	159
1415	55	160
2519	55	161
2700	55	162
1690	55	163
2520	55	164
1319	55	165
1779	55	166
1428	55	167
1368	55	168
2529	55	169
2522	55	170
1441	55	171
2523	55	172
2524	55	173
1562	55	174
2525	55	175
2526	55	176
2534	55	177
2527	55	178
1743	55	179
1358	55	180
1541	55	181
2528	55	182
1416	55	183
2530	55	184
2531	55	185
2532	55	186
1512	55	187
2533	55	188
2549	55	189
1538	55	190
1381	55	191
1338	55	192
2701	55	193
2535	55	194
2536	55	195
2537	55	196
2538	55	197
1783	55	198
2539	55	199
1426	55	200
2540	55	201
2541	55	202
1367	55	203
2542	55	204
1355	55	205
2702	55	206
2564	55	207
2543	55	208
2544	55	209
2545	55	210
1464	55	211
2546	55	212
2547	55	213
2548	55	214
2550	55	215
1546	55	216
2551	55	217
2552	55	218
2553	55	219
2554	55	220
2555	55	221
2556	55	222
2557	55	223
2703	55	224
1405	55	225
1688	55	226
1292	55	227
1324	55	228
2558	55	229
2559	55	230
2704	55	231
2560	55	232
1419	55	233
1361	55	234
2561	55	235
2562	55	236
2563	55	237
1574	55	238
2565	55	239
2566	55	240
2567	55	241
2568	55	242
2569	55	243
2570	55	244
1451	55	245
2571	55	246
2572	55	247
1663	55	248
2573	55	249
2705	55	250
2451	56	1
2452	57	1
2706	58	1
2580	59	1
\.


--
-- Data for Name: publications; Type: TABLE DATA; Schema: public; Owner: taxonomist
--

COPY public.publications (pub_id, pub_title, pub_authors, pub_abstract, pub_eprint, pub_year, pub_url, pub_relevant, pub_citation_count) FROM stdin;
1273	Linguistically-informed self-attention for semantic role labeling	E Strubell, P Verga, D Andor, D Weiss…	The current state-of-the-art end-to-end semantic role labeling (SRL) model is a deep neural network architecture with no explicit linguistic features. However, prior work has shown that gold syntax trees can dramatically improve SRL, suggesting that neural network models …	\N	2018	https://arxiv.org/abs/1804.08199	t	85
558	Recent trends in deep learning based natural language processing	T Young, D Hazarika, S Poria…	Deep learning methods employ multiple processing layers to learn hierarchical representations of data, and have produced state-of-the-art results in many domains. Recently, a variety of model designs and methods have blossomed in the context of natural …\n\nComment\n----------\n\nNo	\N	2018	https://ieeexplore.ieee.org/abstract/document/8416973/	f	563
533	Man is to computer programmer as woman is to homemaker? debiasing word embeddings	T Bolukbasi, KW Chang, JY Zou…	The blind application of machine learning runs the risk of amplifying biases present in data. Such a danger is facing us with word embedding, a popular framework to represent text data as vectors which has been used in many machine learning and natural language …\n\nComment\n----------\n\nContains no use cases of WEMs.	\N	2016	http://papers.nips.cc/paper/6227-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings	f	566
530	word2vec parameter learning explained	X Rong	The word2vec model and application by Mikolov et al. have attracted a great amount of attention in recent two years. The vector representations of words learned by word2vec models have been shown to carry semantic meanings and are useful in various NLP tasks …\n\nComment\n----------\n\nNo use cases, only 	\N	2014	https://arxiv.org/abs/1411.2738	f	451
538	Clevr: A diagnostic dataset for compositional language and elementary visual reasoning	J Johnson, B Hariharan…	When building artificial intelligence systems that can reason and answer questions about visual data, we need diagnostic tests to analyze our progress and discover short-comings. Existing benchmarks for visual question answer-ing can help, but have strong biases that …\n\nComment\n----------\n\nSkip-grams are used in a system for Visual Question Answering, but this system is merely presented as a baseline and not as the real contribution of the paper.	\N	2017	http://openaccess.thecvf.com/content_cvpr_2017/html/Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.html	f	452
551	metapath2vec: Scalable representation learning for heterogeneous networks	Y Dong, NV Chawla, A Swami	We study the problem of representation learning in heterogeneous networks. Its unique challenges come from the existence of multiple types of nodes and links, which limit the feasibility of the conventional network embedding techniques. We develop two scalable …\n\nComment\n----------\n\nNot about WEMs, but about Node Embeddings.	\N	2017	https://dl.acm.org/citation.cfm?id=3098036	f	398
536	Learning end-to-end goal-oriented dialog	A Bordes, YL Boureau, J Weston	Traditional dialog systems used in goal-oriented applications require a lot of domain-specific handcrafting, which hinders scaling up to new domains. End-to-end dialog systems, in which all components are trained from the dialogs themselves, escape this limitation. But the …\n\nComment\n----------\n\nWEMs are used, but only to validate a benchmark.	\N	2016	https://arxiv.org/abs/1605.07683	f	387
534	Show and tell: Lessons learned from the 2015 mscoco image captioning challenge	O Vinyals, A Toshev, S Bengio…	Automatically describing the content of an image is a fundamental problem in artificial intelligence that connects computer vision and natural language processing. In this paper, we present a generative model based on a deep recurrent architecture that combines recent …\n\nComment\n----------\n\nThis is a follow-up of the [Vinyals 2015] paper.	\N	2016	https://ieeexplore.ieee.org/abstract/document/7505636/	f	385
478	Deep learning	I Goodfellow, Y Bengio, A Courville	An introduction to a broad range of topics in deep learning, covering mathematical and conceptual background, deep learning techniques used in industry, and research perspectives.“Written by three experts in the field, Deep Learning is the only comprehensive …\n\nComment\n----------\n\nToo broad.	\N	2016	https://books.google.com/books?hl=en&lr=&id=omivDQAAQBAJ&oi=fnd&pg=PR5&ots=MMT6etnzUR&sig=vZS4Z5p0Dx9Fo5coXQem3CZSvO8	f	12534
1805	Efficient estimation of word representations in vector space	T Mikolov, K Chen, G Corrado, J Dean	We propose two novel model architectures for computing continuous vector representations of words from very large data sets.\nThe quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks.\n\nWe observe **large improvements in accuracy at much lower computational cost**, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set.\nFurthermore, we show that these **vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities.**\n\n\n\nComment\n----------\n\nThis paper does not itself describe many use cases of the model. The quality of the model is assessed with a self-prepared set of semantic and syntactic questions - which seems dangerous w.r.t. overfitting.\nHowever the usefulness of the model follows from the number of citations this has received.\n\nTo find the use cases of Word2Vec we must crawl the citing publications.	\N	2013	https://arxiv.org/abs/1301.3781	t	14011
480	Tensorflow: A system for large-scale machine learning	M Abadi, P Barham, J Chen, Z Chen, A Davis…	TensorFlow is a machine learning system that operates at large scale and in heterogeneous environments. Tensor-Flow uses dataflow graphs to represent computation, shared state, and the operations that mutate that state. It maps the nodes of a dataflow graph across many …\n\nComment\n----------\n\nDoes not cite Mikolov :(	\N	2016	https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi	f	6664
495	Knowledge vault: A web-scale approach to probabilistic knowledge fusion	X Dong, E Gabrilovich, G Heitz, W Horn, N Lao…	Recent years have witnessed a proliferation of large-scale knowledge bases, including Wikipedia, Freebase, YAGO, Microsoft's Satori, and Google's Knowledge Graph. To increase the scale even further, we need to explore automatic methods for constructing knowledge …	\N	2014	https://dl.acm.org/citation.cfm?id=2623623	f	1040
554	Deep graph kernels	P Yanardag, SVN Vishwanathan	In this paper, we present Deep Graph Kernels, a unified framework to learn latent representations of sub-structures for graphs, inspired by latest advancements in language modeling and deep learning. Our framework leverages the dependency information …	\N	2015	https://dl.acm.org/citation.cfm?id=2783417	t	308
1271	Quac: Question answering in context	E Choi, H He, M Iyyer, M Yatskar, W Yih, Y Choi…	We present QuAC, a dataset for Question Answering in Context that contains 14K information-seeking QA dialogs (100K questions in total). The interactions involve two crowd workers:(1) a student who poses a sequence of freeform questions to learn as much as …	\N	2018	https://arxiv.org/abs/1808.07036	f	73
1348	An overview of deep learning in medical imaging focusing on MRI	AS Lundervold, A Lundervold	What has happened in machine learning lately, and what does it mean for the future of medical image analysis? Machine learning has witnessed a tremendous amount of attention over the last few years. The current boom started around 2009 when so-called deep artificial …	\N	2019	https://www.sciencedirect.com/science/article/pii/S0939388918301181	t	63
1263	Improving language understanding by generative pre-training	A Radford, K Narasimhan, T Salimans…	Natural language understanding comprises a wide range of diverse tasks such as textual entailment, question answering, semantic similarity assessment, and document classification.\n\nAlthough large unlabeled text corpora are abundant, labeled data for learning these specific tasks is scarce, making it challenging for discriminatively trained models to perform adequately.\n\nWe demonstrate that large gains on these tasks can be realized by **generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task.**\n\nIn contrast to previous approaches, we make use of **task-aware input transformations during fine-tuning** to achieve effective transfer while requiring minimal changes to the model architecture.\n\nWe demonstrate the effectiveness of our approach on a wide range of benchmarks for natural language understanding.\n\nOur general task-agnostic model outperforms discriminatively trained models that use architectures specifically crafted for each task, significantly improving upon the state of the art in 9 out of the 12 tasks studied.\n\nFor instance, we achieve absolute improvements of\n\n - 8.9% on commonsense reasoning (Stories Cloze Test),\n - 5.7% on question answering (RACE), and\n - 1.5% on textual entailment (MultiNLI).	\N	2018	https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf	f	534
1267	Glue: A multi-task benchmark and analysis platform for natural language understanding	A Wang, A Singh, J Michael, F Hill, O Levy…	For natural language understanding (NLU) technology to be maximally useful, both practically and as a scientific object of study, it must be general: it must be able to process language in a way that is not exclusively tailored to any one specific task or dataset. In …	\N	2018	https://arxiv.org/abs/1804.07461	t	255
1354	Gender Bias in Contextualized Word Embeddings	J Zhao, T Wang, M Yatskar, R Cotterell…	In this paper, we quantify, analyze and mitigate gender bias exhibited in ELMo's contextualized word vectors. First, we conduct several intrinsic analyses and find that (1) training data for ELMo contains significantly more male than female entities,(2) the trained …	\N	2019	https://arxiv.org/abs/1904.03310	t	24
506	Unifying visual-semantic embeddings with multimodal neural language models	R Kiros, R Salakhutdinov, RS Zemel	Inspired by recent advances in multimodal learning and machine translation, we introduce an encoder-decoder pipeline that learns\n\n 1. a multimodal joint embedding space with images and text and\n 2. a novel language model for decoding distributed representations from our space.\n\nOur pipeline effectively **unifies joint image-text embedding models with multimodal neural language models**. We introduce the structure-content neural language model that disentangles the structure of a sentence to its content, conditioned on representations produced by the encoder.\nThe encoder allows one to rank images and sentences while the decoder can generate novel descriptions from scratch. Using LSTM to encode sentences, we **match the state-of-the-art performance on Flickr8K and Flickr30K without using object detections**. We also set new best results when using the 19-layer Oxford convolutional network. Furthermore we show that with linear encoders, the **learned embedding space captures multimodal regularities in terms of vector space arithmetic e.g. *image of a blue car* - "blue" + "red" is near images of red cars**.\nSample captions generated for 800 images are made available for comparison.	\N	2014	https://arxiv.org/abs/1411.2539	t	739
1369	Videobert: A joint model for video and language representation learning	C Sun, A Myers, C Vondrick, K Murphy…	Self-supervised learning has become increasingly important to leverage the abundance of unlabeled data available on platforms like YouTube. Whereas most existing approaches learn low-level representations, we propose a joint visual-linguistic model to learn high-level …	\N	2019	https://arxiv.org/abs/1904.01766	t	28
555	Simple and scalable predictive uncertainty estimation using deep ensembles	B Lakshminarayanan, A Pritzel…	Deep neural networks (NNs) are powerful black box predictors that have recently achieved impressive performance on a wide spectrum of tasks. Quantifying predictive uncertainty in NNs is a challenging and yet unsolved problem. Bayesian NNs, which learn a distribution …\n\nComment\n----------\n\nNot about WEMs.	\N	2017	http://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles	f	482
529	Deep sentence embedding using long short-term memory networks: Analysis and application to information retrieval	H Palangi, L Deng, Y Shen, J Gao, X He…	This paper develops a model that addresses sentence embedding, a hot topic in current natural language processing research, using recurrent neural networks (RNN) with Long Short-Term Memory (LSTM) cells. The proposed LSTM-RNN model sequentially takes each …	\N	2016	https://dl.acm.org/citation.cfm?id=2992457	t	405
527	Exploring models and data for image question answering	M Ren, R Kiros, R Zemel	This work aims to address the problem of image-based question-answering (QA) with new models and datasets. In our work, we propose to use neural networks and visual semantic embeddings, without intermediate stages such as object detection and image segmentation …	\N	2015	http://papers.nips.cc/paper/5640-exploring-models-and-data-for-image-question-answering	t	404
1385	Publicly available clinical BERT embeddings	E Alsentzer, JR Murphy, W Boag, WH Weng…	Contextual word embedding models such as ELMo (Peters et al., 2018) and BERT (Devlin et al., 2018) have dramatically improved performance for many natural language processing (NLP) tasks in recent months. However, these models have been minimally explored on …	\N	2019	https://arxiv.org/abs/1904.03323	t	34
528	Linguistic regularities in sparse and explicit word representations	O Levy, Y Goldberg	Recent work has shown that neural embedded word representations capture many relational similarities, which can be recovered by means of vector arithmetic in the embedded space. We show that Mikolov et al.'s method of first adding and subtracting word vectors, and then …	\N	2014	https://www.aclweb.org/anthology/W14-1618	t	385
547	Aligning books and movies: Towards story-like visual explanations by watching movies and reading books	Y Zhu, R Kiros, R Zemel, R Salakhutdinov…	Books are a rich source of both fine-grained information, how a character, an object or a scene looks like, as well as high-level semantics, what someone is thinking, feeling and how these states evolve through a story. This paper aims to align books to their movie releases in …	\N	2015	https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zhu_Aligning_Books_and_ICCV_2015_paper.html	t	376
553	Opportunities and obstacles for deep learning in biology and medicine	T Ching, DS Himmelstein…	Deep learning describes a class of machine learning algorithms that are capable of combining raw inputs into layers of intermediate features. These algorithms have recently shown impressive results across a variety of domains. Biology and medicine are data-rich …	\N	2018	https://royalsocietypublishing.org/doi/abs/10.1098/rsif.2017.0387	t	363
532	Mind's eye: A recurrent visual representation for image caption generation	X Chen, C Lawrence Zitnick	In this paper we explore the bi-directional mapping between images and their sentence-based descriptions. Critical to our approach is a recurrent neural network that attempts to dynamically build a visual representation of the scene as a caption is being generated or …	\N	2015	https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Chen_Minds_Eye_A_2015_CVPR_paper.html	t	347
557	Learning from class-imbalanced data: Review of methods and applications	G Haixiang, L Yijing, J Shang, G Mingyun…	Rare events, especially those that could potentially negatively impact society, often require humans' decision-making responses. Detecting rare events can be viewed as a prediction task in data mining and machine learning communities. As these events are rarely observed …	\N	2017	https://www.sciencedirect.com/science/article/pii/S0957417416307175	t	345
544	Synthesized classifiers for zero-shot learning	S Changpinyo, WL Chao, B Gong…	Given semantic descriptions of object classes, zero-shot learning aims to accurately recognize objects of the unseen classes, from which no examples are available at the training stage, by associating them to the seen classes, from which labeled examples are …	\N	2016	https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Changpinyo_Synthesized_Classifiers_for_CVPR_2016_paper.html	t	344
549	A review of affective computing: From unimodal analysis to multimodal fusion	S Poria, E Cambria, R Bajpai, A Hussain	Affective computing is an emerging interdisciplinary research field bringing together researchers and practitioners from various fields, ranging from artificial intelligence, natural language processing, to cognitive and social sciences. With the proliferation of videos …	\N	2017	https://www.sciencedirect.com/science/article/pii/S1566253517300738	t	342
550	Abusive language detection in online user content	C Nobata, J Tetreault, A Thomas, Y Mehdad…	Detection of abusive language in user generated online content has become an issue of increasing importance in recent years. Most current commercial methods make use of blacklists and regular expressions, however these measures fall short when contending with …	\N	2016	https://dl.acm.org/citation.cfm?id=2883062	t	333
556	Visual relationship detection with language priors	C Lu, R Krishna, M Bernstein, L Fei-Fei	Visual relationships capture a wide variety of interactions between pairs of objects in images (eg “man riding bicycle” and “man pushing bicycle”). Consequently, the set of possible relationships is extremely large and it is difficult to obtain sufficient training examples for all …	\N	2016	https://link.springer.com/chapter/10.1007/978-3-319-46448-0_51	t	333
562	Deep reinforcement learning: An overview	Y Li	We give an overview of recent exciting achievements of deep reinforcement learning (RL). We discuss six core elements, six important mechanisms, and twelve applications. We start with background of machine learning, deep learning and reinforcement learning. Next we …	\N	2017	https://arxiv.org/abs/1701.07274	t	329
2482	Gpipe: Efficient training of giant neural networks using pipeline parallelism	Y Huang, Y Cheng, A Bapna, O Firat…	Scaling up deep neural network capacity has been known as an effective approach to improving model quality for several different machine learning tasks. In many cases, increasing model capacity beyond the memory limit of a single accelerator has required …\n\nDoes not cite ELMo :(	\N	2019	http://papers.nips.cc/paper/8305-gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism	f	85
535	Efficient non-parametric estimation of multiple embeddings per word in vector space	A Neelakantan, J Shankar, A Passos…	There is rising interest in vector-space word embeddings and their use in NLP, especially given recent methods for their fast estimation at very large scale. Nearly all this work, however, assumes a single vector per word type ignoring polysemy and thus jeopardizing …	\N	2015	https://arxiv.org/abs/1504.06654	t	325
1284	Bottom-up abstractive summarization	S Gehrmann, Y Deng, AM Rush	Neural network-based methods for abstractive summarization produce outputs that are more fluent than other techniques, but which can be poor at content selection. This work proposes a simple technique for addressing this issue: use a data-efficient content selector to over …	\N	2018	https://arxiv.org/abs/1808.10792	t	83
540	Classifying relations by ranking with convolutional neural networks	CN Santos, B Xiang, B Zhou	Relation classification is an important semantic processing task for which state-ofthe-art systems still rely on costly handcrafted features. In this work we tackle the relation classification task using a convolutional neural network that performs classification by …	\N	2015	https://arxiv.org/abs/1504.06580	t	317
1282	Swag: A large-scale adversarial dataset for grounded commonsense inference	R Zellers, Y Bisk, R Schwartz, Y Choi	Given a partial description like" she opened the hood of the car," humans can reason about the situation and anticipate what might come next (" then, she examined the engine"). In this paper, we introduce the task of grounded commonsense inference, unifying natural …	\N	2018	https://arxiv.org/abs/1808.05326	t	79
568	Recurrent neural network for text classification with multi-task learning	P Liu, X Qiu, X Huang	Neural network based methods have obtained great progress on a variety of natural language processing tasks. However, in most previous works, the models are learned based on single-task supervised objectives, which often suffer from insufficient training data. In this …	\N	2016	https://arxiv.org/abs/1605.05101	t	312
542	Pharmacovigilance from social media: mining adverse drug reaction mentions using sequence labeling with word embedding cluster features	A Nikfarjam, A Sarker, K O'Connor…	Objective Social media is becoming increasingly popular as a platform for sharing personal health-related information. This information can be utilized for public health monitoring tasks, particularly for pharmacovigilance, via the use of natural language processing (NLP) …	\N	2015	https://academic.oup.com/jamia/article-abstract/22/3/671/776531	t	308
584	Deep learning: A critical appraisal	G Marcus	Although deep learning has historical roots going back decades, neither the term" deep learning" nor the approach was popular just over five years ago, when the field was reignited by papers such as Krizhevsky, Sutskever and Hinton's now classic (2012) deep …	\N	2018	https://arxiv.org/abs/1801.00631	t	307
664	Computational optimal transport	G Peyré, M Cuturi	Optimal transport (OT) theory can be informally described using the words of the French mathematician Gaspard Monge (1746–1818): A worker with a shovel in hand has to move a large pile of sand lying on a construction site. The goal of the worker is to erect with all that …	\N	2019	http://www.nowpublishers.com/article/Details/MAL-073	t	305
537	A neural network for factoid question answering over paragraphs	M Iyyer, J Boyd-Graber, L Claudino, R Socher…	Text classification methods for tasks like factoid question answering typically use manually defined string matching rules or bag of words representations. These methods are ineffective when question text contains very few individual words (eg, named entities) that …	\N	2014	https://www.aclweb.org/anthology/D14-1070	t	296
578	Community preserving network embedding	X Wang, P Cui, J Wang, J Pei, W Zhu, S Yang	Network embedding, aiming to learn the low-dimensional representations of nodes in networks, is of paramount importance in many real applications. One basic requirement of network embedding is to preserve the structure and inherent properties of the networks …	\N	2017	https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14589	t	295
1307	Xnli: Evaluating cross-lingual sentence representations	A Conneau, G Lample, R Rinott, A Williams…	State-of-the-art natural language processing systems rely on supervision in the form of annotated data to learn competent models. These models are generally trained on data in a single language (usually English), and cannot be directly used beyond that language. Since …	\N	2018	https://arxiv.org/abs/1809.05053	t	67
493	Neural word embedding as implicit matrix factorization	O Levy, Y Goldberg	We analyze skip-gram with negative-sampling (SGNS), a word embedding method introduced by Mikolov et al., and show that it is implicitly factorizing a word-context matrix, whose cells are the pointwise mutual information (PMI) of the respective word and context …\n\nComment\n----------\n\nNo use cases, but mathematical details.	\N	2014	http://papers.nips.cc/paper/5477-neural-word-embedding-as	f	1166
565	Deep learning for event-driven stock prediction	X Ding, Y Zhang, T Liu, J Duan	We propose a deep learning method for event-driven stock market prediction. First, events are extracted from news text, and represented as dense vectors, trained using a novel neural tensor network. Second, a deep convolutional neural network is used to model both …	\N	2015	https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/viewPaper/11031	t	275
559	Learning distributed representations of sentences from unlabelled data	F Hill, K Cho, A Korhonen	Unsupervised methods for learning distributed representations of words are ubiquitous in today's NLP research, but far less is known about the best ways to learn distributed phrase or sentence representations from unlabelled data. This paper is a systematic comparison of …	\N	2016	https://arxiv.org/abs/1602.03483	t	274
539	Tailoring continuous word representations for dependency parsing	M Bansal, K Gimpel, K Livescu	Word representations have proven useful for many NLP tasks, eg, Brown clusters as features in dependency parsing (Koo et al., 2008). In this paper, we investigate the use of continuous word representations as features for dependency parsing. We compare several …	\N	2014	https://www.aclweb.org/anthology/P14-2131	t	271
543	Explain images with multimodal recurrent neural networks	J Mao, W Xu, Y Yang, J Wang, AL Yuille	In this paper, we present a multimodal Recurrent Neural Network (m-RNN) model for generating novel sentence descriptions to explain the content of images. It directly models the probability distribution of generating a word given previous words and the image. Image …	\N	2014	https://arxiv.org/abs/1410.1090	t	266
582	Using recurrent neural network models for early detection of heart failure onset	E Choi, A Schuetz, WF Stewart…	Objective: We explored whether use of deep learning to model temporal relations among events in electronic health records (EHRs) would improve model performance in predicting initial diagnosis of heart failure (HF) compared to conventional methods that ignore …	\N	2016	https://academic.oup.com/jamia/article-abstract/24/2/361/2631499	t	266
622	struc2vec: Learning node representations from structural identity	LFR Ribeiro, PHP Saverese…	Structural identity is a concept of symmetry in which network nodes are identified according to the network structure and their relationship to other nodes. Structural identity has been studied in theory and practice over the past decades, but only recently has it been …	\N	2017	https://dl.acm.org/citation.cfm?id=3098061	t	259
1272	Constituency parsing with a self-attentive encoder	N Kitaev, D Klein	We demonstrate that replacing an LSTM encoder with a self-attentive architecture can lead to improvements to a state-of-the-art discriminative constituency parser. The use of attention makes explicit the manner in which information is propagated between different locations in …	\N	2018	https://arxiv.org/abs/1805.01052	t	62
560	Where to look: Focus regions for visual question answering	KJ Shih, S Singh, D Hoiem	We present a method that learns to answer visual questions by selecting image regions relevant to the text-based query. Our method maps textual queries and visual features from various regions into a shared space where they are compared for relevance with an inner …	\N	2016	http://openaccess.thecvf.com/content_cvpr_2016/html/Shih_Where_to_Look_CVPR_2016_paper.html	t	258
567	Document embedding with paragraph vectors	AM Dai, C Olah, QV Le	Paragraph Vectors has been recently proposed as an unsupervised method for learning distributed representations for pieces of texts. In their work, the authors showed that the method can learn an embedding of movie review texts which can be leveraged for sentiment …	\N	2015	https://arxiv.org/abs/1507.07998	t	258
623	Semantic autoencoder for zero-shot learning	E Kodirov, T Xiang, S Gong	Existing zero-shot learning (ZSL) models typically learn a projection function from a feature space to a semantic embedding space (eg attribute space). However, such a projection function is only concerned with predicting the training seen class semantic representation …	\N	2017	http://openaccess.thecvf.com/content_cvpr_2017/html/Kodirov_Semantic_Autoencoder_for_CVPR_2017_paper.html	t	253
616	" liar, liar pants on fire": A new benchmark dataset for fake news detection	WY Wang	Automatic fake news detection is a challenging problem in deception detection, and it has tremendous real-world political and social impacts. However, statistical approaches to combating fake news has been dramatically limited by the lack of labeled benchmark …	\N	2017	https://arxiv.org/abs/1705.00648	t	251
561	Short text similarity with word embeddings	T Kenter, M De Rijke	Determining semantic similarity between texts is important in many tasks in information retrieval such as search, query suggestion, automatic summarization and image finding. Many approaches have been suggested, based on lexical matching, handcrafted patterns …	\N	2015	https://dl.acm.org/citation.cfm?id=2806475	t	259
1294	Deep learning in medical imaging and radiation therapy	B Sahiner, A Pezeshk, LM Hadjiiski, X Wang…	The goals of this review paper on deep learning (DL) in medical imaging and radiation therapy are to (a) summarize what has been achieved to date;(b) identify common and unique challenges, and strategies that researchers have taken to address these challenges; …	\N	2019	https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.13264	t	62
1300	Semi-supervised sequence modeling with cross-view training	K Clark, MT Luong, CD Manning, QV Le	Unsupervised representation learning algorithms such as word2vec and ELMo improve the accuracy of many supervised NLP models, mainly because they can take advantage of large amounts of unlabeled text. However, the supervised models only learn from task-specific …	\N	2018	https://arxiv.org/abs/1809.08370	t	61
546	Multilingual models for compositional distributed semantics	KM Hermann, P Blunsom	We present a novel technique for learning semantic representations, which extends the distributional hypothesis to multilingual data and joint-space embeddings.\nOur models leverage parallel data and learn to strongly **align the embeddings of semantically equivalent sentences, while maintaining sufficient distance between those of dissimilar sentences**. The models do not rely on word alignments or any syntactic information and are successfully applied to a number of diverse languages. We extend our approach to learn **semantic representations at the document level, too**.\n\nWe evaluate these models on two **cross-lingual document classification** tasks, outperforming the prior state of the art. Through qualitative analysis and the study of pivoting effects we demonstrate that our representations are semantically plausible and can capture semantic relationships across languages without parallel data.	\N	2014	https://arxiv.org/abs/1404.4641	t	247
548	A unified model for word sense representation and disambiguation	X Chen, Z Liu, M Sun	Most word representation methods assume that each word owns a single semantic vector. This is usually problematic because lexical ambiguity is ubiquitous, which is also the problem to be resolved by word sense disambiguation. In this paper, we present a unified …	\N	2014	https://www.aclweb.org/anthology/D14-1110	t	247
566	A hierarchical recurrent encoder-decoder for generative context-aware query suggestion	A Sordoni, Y Bengio, H Vahabi, C Lioma…	Users may strive to formulate an adequate textual query for their information need. Search engines assist the users by presenting query suggestions. To preserve the original search intent, suggestions should be context-aware and account for the previous queries issued by …	\N	2015	https://dl.acm.org/citation.cfm?id=2806493	t	245
592	Distributional semantics resources for biomedical text processing	S Moen, TSS Ananiadou	The openly available biomedical literature contains over 5 billion words in publication abstracts and full texts. Recent advances in unsupervised language processing methods have made it possible to make use of such large unannotated corpora for building statistical …	\N	2013	https://pdfs.semanticscholar.org/e2f2/8568031e1902d4f8ee818261f0f2c20de6dd.pdf	t	240
564	Deep convolutional neural network textual features and multiple kernel learning for utterance-level multimodal sentiment analysis	S Poria, E Cambria, A Gelbukh	We present a novel way of extracting features from short texts, based on the activation values of an inner layer of a deep convolutional neural network. We use the extracted features in multimodal sentiment analysis of short video clips representing one sentence …	\N	2015	https://www.aclweb.org/anthology/D15-1303	t	235
581	Movieqa: Understanding stories in movies through question-answering	M Tapaswi, Y Zhu, R Stiefelhagen…	We introduce the MovieQA dataset which aims to evaluate automatic story comprehension from both video and text. The dataset consists of 14,944 questions about 408 movies with high semantic diversity. The questions range from simpler" Who" did" What" to" Whom", to" …	\N	2016	https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Tapaswi_MovieQA_Understanding_Stories_CVPR_2016_paper.html	t	235
632	Deep EHR: a survey of recent advances in deep learning techniques for electronic health record (EHR) analysis	B Shickel, PJ Tighe, A Bihorac…	The past decade has seen an explosion in the amount of digital information stored in electronic health records (EHRs). While primarily designed for archiving patient information and performing administrative healthcare tasks like billing, many researchers have found …	\N	2017	https://ieeexplore.ieee.org/abstract/document/8086133/	t	235
1278	From word to sense embeddings: A survey on vector representations of meaning	J Camacho-Collados, MT Pilehvar	Over the past years, distributed semantic representations have proved to be effective and flexible keepers of prior knowledge to be integrated into downstream applications. This survey focuses on the representation of meaning. We start from the theoretical background …	\N	2018	https://www.jair.org/index.php/jair/article/view/11259	t	59
638	A survey on hate speech detection using natural language processing	A Schmidt, M Wiegand	This paper presents a survey on hate speech detection. Given the steadily growing body of social media content, the amount of online hate speech is also increasing. Due to the massive scale of the web, methods that automatically detect hate speech are required. Our …	\N	2017	https://www.aclweb.org/anthology/papers/W/W17/W17-1101/	t	233
563	Semeval-2015 task 2: Semantic textual similarity, english, spanish and pilot on interpretability	E Agirre, C Banea, C Cardie, D Cer, M Diab…	In semantic textual similarity (STS), systems rate the degree of semantic equivalence between two text snippets. This year, the participants were challenged with new datasets in English and Spanish. The annotations for both subtasks leveraged crowdsourcing. The …	\N	2015	https://www.aclweb.org/anthology/S15-2045	t	228
575	Unsupervised domain adaptation for zero-shot learning	E Kodirov, T Xiang, Z Fu, S Gong	Zero-shot learning (ZSL) can be considered as a special case of transfer learning where the source and target domains have different tasks/label spaces and the target domain is unlabelled, providing little guidance for the knowledge transfer. A ZSL method typically …	\N	2015	https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Kodirov_Unsupervised_Domain_Adaptation_ICCV_2015_paper.html	t	221
573	Transductive multi-view zero-shot learning	Y Fu, TM Hospedales, T Xiang…	Most existing zero-shot learning approaches exploit transfer learning via an intermediate semantic representation shared between an annotated auxiliary dataset and a target dataset with different classes and no annotation. A projection from a low-level feature space to the …	\N	2015	https://ieeexplore.ieee.org/abstract/document/7053935/	t	217
678	Generative adversarial networks: An overview	A Creswell, T White, V Dumoulin…	Generative adversarial networks (GANs) provide a way to learn deep representations without extensively annotated training data. They achieve this by deriving backpropagation signals through a competitive process involving a pair of networks. The representations that …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8253599/	t	216
579	Statistically significant detection of linguistic change	V Kulkarni, R Al-Rfou, B Perozzi, S Skiena	We propose a new computational approach for tracking and detecting statistically significant linguistic shifts in the meaning and usage of words. Such linguistic shifts are especially prevalent on the Internet, where the rapid exchange of ideas can quickly change a word's …	\N	2015	https://dl.acm.org/citation.cfm?id=2741627	t	214
576	Using the output embedding to improve language models	O Press, L Wolf	We study the topmost weight matrix of neural network language models. We show that this matrix constitutes a valid word embedding. When training language models, we recommend tying the input embedding and this output embedding. We analyze the resulting update rules …	\N	2016	https://arxiv.org/abs/1608.05859	t	252
596	Argumentation mining: State of the art and emerging trends	M Lippi, P Torroni	Argumentation mining aims at automatically extracting structured arguments from unstructured textual documents. It has recently become a hot topic also due to its potential in processing information originating from the Web, and in particular from social media, in …	\N	2016	https://dl.acm.org/citation.cfm?id=2850417	t	214
1280	Higher-order coreference resolution with coarse-to-fine inference	K Lee, L He, L Zettlemoyer	We introduce a fully differentiable approximation to higher-order inference for coreference resolution. Our approach uses the antecedent distribution from a span-ranking architecture as an attention mechanism to iteratively refine span representations. This enables the model …	\N	2018	https://arxiv.org/abs/1804.05392	t	55
1283	Multi-granularity hierarchical attention fusion networks for reading comprehension and question answering	W Wang, M Yan, C Wu	This paper describes a novel hierarchical attention network for reading comprehension style question answering, which aims to answer questions for a given narrative paragraph. In the proposed method, attention and fusion are conducted horizontally and vertically across …	\N	2018	https://arxiv.org/abs/1811.11934	t	51
569	Lexicon infused phrase embeddings for named entity resolution	A Passos, V Kumar, A McCallum	Most state-of-the-art approaches for named-entity recognition (NER) use semi supervised information in the form of word clusters and lexicons. Recently neural network-based language models have been explored, as they as a byproduct generate highly informative …	\N	2014	https://arxiv.org/abs/1404.5367	t	209
585	Multimodal convolutional neural networks for matching image and sentence	L Ma, Z Lu, L Shang, H Li	In this paper, we propose multimodal convolutional neural networks (m-CNNs) for matching image and sentence. Our m-CNN provides an end-to-end framework with convolutional architectures to exploit image representation, word composition, and the matching relations …	\N	2015	http://openaccess.thecvf.com/content_iccv_2015/html/Ma_Multimodal_Convolutional_Neural_ICCV_2015_paper.html	t	205
1309	Learning gender-neutral word embeddings	J Zhao, Y Zhou, Z Li, W Wang, KW Chang	Word embedding models have become a fundamental component in a wide range of Natural Language Processing (NLP) applications. However, embeddings trained on human-generated corpora have been demonstrated to inherit strong gender stereotypes that reflect …	\N	2018	https://arxiv.org/abs/1809.01496	t	49
601	Tri-party deep network representation	S Pan, J Wu, X Zhu, C Zhang, Y Wang	Information network mining often requires examination of linkage relationships between nodes for analysis. Recently, network representation has emerged to represent each node in a vector format, embedding network structure, so off-the-shelf machine …	\N	2016	https://www.researchgate.net/profile/Yang_Wang81/publication/305441156_Tri-Party_Deep_Network_Representation/links/57906bbb08ae4e917cff4119.pdf	t	203
552	Evaluating the performance of four snooping cache coherency protocols	SJ Eggers, RH Katz	Write-invalidate and write-broadcast coherency protocols have been criticized for being unable to achieve good bus performance across all cache configurations. In particular, write-invalidate performance can suffer as block size increases; and large cache sizes will hurt …	\N	1989	https://dl.acm.org/citation.cfm?id=74927	t	202
583	Sensembed: Learning sense embeddings for word and relational similarity	I Iacobacci, MT Pilehvar, R Navigli	Word embeddings have recently gained considerable popularity for modeling words in different Natural Language Processing (NLP) tasks including semantic similarity measurement. However, notwithstanding their success, word embeddings are by their very …	\N	2015	https://www.aclweb.org/anthology/P15-1010	t	201
1277	Semantic sentence matching with densely-connected recurrent and co-attentive information	S Kim, I Kang, N Kwak	Sentence matching is widely used in various natural language tasks such as natural language inference, paraphrase identification, and question answering. For these tasks, understanding logical and semantic relationship between two sentences is required but it is …	\N	2019	https://www.aaai.org/ojs/index.php/AAAI/article/view/4627	t	48
605	Chinese comments sentiment classification based on word2vec and SVMperf	D Zhang, H Xu, Z Su, Y Xu	Since the booming development of e-commerce in the last decade, the researchers have begun to pay more attention to extract the valuable information from consumers comments. Sentiment classification, which focuses on classify the comments into positive class and …	\N	2015	https://www.sciencedirect.com/science/article/pii/S0957417414005508	t	200
595	How to generate a good word embedding	S Lai, K Liu, S He, J Zhao	The authors analyze three critical components in training word embeddings: model, corpus, and training parameters. They systematize existing neural-network-based word embedding methods and experimentally compare them using the same corpus. They then evaluate …	\N	2016	https://ieeexplore.ieee.org/abstract/document/7478417/	t	214
1290	Techniques for interpretable machine learning	M Du, N Liu, X Hu	Interpretable machine learning tackles the important problem that humans cannot understand the behaviors of complex machine learning models and how these classifiers arrive at a particular decision. Although many approaches have been proposed, a …	\N	2018	https://arxiv.org/abs/1808.00033	t	47
1276	Jointly predicting predicates and arguments in neural semantic role labeling	L He, K Lee, O Levy, L Zettlemoyer	Recent BIO-tagging-based neural semantic role labeling models are very high performing, but assume gold predicates as part of the input and cannot incorporate span-level features. We propose an end-to-end approach for jointly predicting all predicates, arguments spans …	\N	2018	https://arxiv.org/abs/1805.04787	t	46
613	Shuffle and learn: unsupervised learning using temporal order verification	I Misra, CL Zitnick, M Hebert	In this paper, we present an approach for learning a visual representation from the raw spatiotemporal signals in videos. Our representation is learned without supervision from semantic labels. We formulate our method as an unsupervised sequential verification task …	\N	2016	https://link.springer.com/chapter/10.1007/978-3-319-46448-0_32	t	199
1295	A simple method for commonsense reasoning	TH Trinh, QV Le	Commonsense reasoning is a long-standing challenge for deep learning. For example, it is difficult to use neural networks to tackle the Winograd Schema dataset~\\cite {levesque2011winograd}. In this paper, we present a simple method for commonsense …	\N	2018	https://arxiv.org/abs/1806.02847	t	46
1398	From Recognition to Cognition: Visual Commonsense Reasoning	R Zellers, Y Bisk, A Farhadi…	Visual understanding goes well beyond object recognition. With one glance at an image, we can effortlessly imagine the world beyond the pixels: for instance, we can infer people's actions, goals, and mental states. While this task is easy for humans, it is tremendously …	\N	2019	http://openaccess.thecvf.com/content_CVPR_2019/html/Zellers_From_Recognition_to_Cognition_Visual_Commonsense_Reasoning_CVPR_2019_paper.html	t	44
574	Compositional morphology for word representations and language modelling	J Botha, P Blunsom	This paper presents a scalable method for integrating compositional morphological representations into a vector-based probabilistic language model. Our approach is evaluated in the context of log-bilinear language models, rendered suitably efficient for …	\N	2014	http://www.jmlr.org/proceedings/papers/v32/botha14.pdf	t	196
586	Learning natural coding conventions	M Allamanis, ET Barr, C Bird, C Sutton	Every programmer has a characteristic style, ranging from preferences about identifier naming to preferences about object relationships and design patterns. Coding conventions define a consistent syntactic style, fostering readability and hence maintainability. When …	\N	2014	https://dl.acm.org/citation.cfm?id=2635883	t	195
594	Boosting named entity recognition with neural character embeddings	CN Santos, V Guimaraes	Most state-of-the-art named entity recognition (NER) systems rely on handcrafted features and on the output of other NLP tasks such as part-of-speech (POS) tagging and text chunking. In this work we propose a language-independent NER system that uses …	\N	2015	https://arxiv.org/abs/1505.05008	t	195
1408	A structural probe for finding syntax in word representations	J Hewitt, CD Manning	Recent work has improved our ability to detect linguistic knowledge in word representations. However, current methods for detecting syntactic knowledge do not test whether syntax trees are represented in their entirety. In this work, we propose a structural probe, which evaluates …	\N	2019	https://www.aclweb.org/anthology/N19-1419.pdf	t	43
620	Rdf2vec: Rdf graph embeddings for data mining	P Ristoski, H Paulheim	Linked Open Data has been recognized as a valuable source for background information in data mining. However, most data mining tools require features in propositional form, ie, a vector of nominal or numerical features associated with an instance, while Linked …	\N	2016	https://link.springer.com/chapter/10.1007/978-3-319-46523-4_30	t	193
612	Convolutional MKL based multimodal emotion recognition and sentiment analysis	S Poria, I Chaturvedi, E Cambria…	Technology has enabled anyone with an Internet connection to easily create and share their ideas, opinions and content with millions of other people around the world. Much of the content being posted and consumed online is multimodal. With billions of phones, tablets …	\N	2016	https://ieeexplore.ieee.org/abstract/document/7837868/	t	192
609	Improving zero-shot learning by mitigating the hubness problem	G Dinu, A Lazaridou, M Baroni	The zero-shot paradigm exploits vector-based word representations extracted from text corpora with unsupervised methods to learn general mapping functions from other feature spaces onto word space, where the words associated to the nearest neighbours of the …	\N	2014	https://arxiv.org/abs/1412.6568	t	188
590	Learning semantic hierarchies via word embeddings	R Fu, J Guo, B Qin, W Che, H Wang, T Liu	Semantic hierarchy construction aims to build structures of concepts linked by hypernym–hyponym (“is-a”) relations. A major challenge for this task is the automatic discovery of such relations. This paper proposes a novel and effective method for the construction of semantic …	\N	2014	https://www.aclweb.org/anthology/P14-1113	t	187
630	Text matching as image recognition	L Pang, Y Lan, J Guo, J Xu, S Wan, X Cheng	Matching two texts is a fundamental problem in many natural language processing tasks. An effective way is to extract meaningful matching patterns from words, phrases, and sentences to produce the matching score. Inspired by the success of convolutional neural network in …	\N	2016	https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/11895	t	186
675	Learning a deep embedding model for zero-shot learning	L Zhang, T Xiang, S Gong	Zero-shot learning (ZSL) models rely on learning a joint embedding space where both textual/semantic description of object classes and visual representation of object images can be projected to for nearest neighbour search. Despite the success of deep neural networks …	\N	2017	http://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Learning_a_Deep_CVPR_2017_paper.html	t	184
1483	Coarse-grain Fine-grain Coattention Network for Multi-evidence Question Answering	V Zhong, C Xiong, NS Keskar, R Socher	End-to-end neural models have made significant progress in question answering, however recent studies show that these models implicitly assume that the answer and evidence appear close together in a single document. In this work, we propose the Coarse-grain Fine …	\N	2019	https://arxiv.org/abs/1901.00603	t	12
570	Light emitting device	M Azami, Y Tanada	US PATENT DOCUMENTS 6,501,098 B2 12/2002 Yamazaki 6,522,323 B1 2/2003 Sasaki et al. 4,090,096 A 5/1978 Nagami 6,535,185 B2 3/2003 Kim et al..................... 345/76 4,390,803 A 6/1983 Koike 6,542,138 B1 4/2003 Shannon et al............... 345/76 4,412,139 A 10/1983 …	\N	2005	https://patents.google.com/patent/US6958750B2/en	t	181
625	Item2vec: neural item embedding for collaborative filtering	O Barkan, N Koenigstein	Many Collaborative Filtering (CF) algorithms are item-based in the sense that they analyze item-item relations in order to produce item similarities. Recently, several works in the field of Natural Language Processing (NLP) suggested to learn a latent representation of words …	\N	2016	https://ieeexplore.ieee.org/abstract/document/7738886/	t	181
1496	End-to-End Open-Domain Question Answering with BERTserini	W Yang, Y Xie, A Lin, X Li, L Tan, K Xiong, M Li…	We demonstrate an end-to-end question answering system that integrates BERT with the open-source Anserini information retrieval toolkit. In contrast to most question answering and reading comprehension models today, which operate over small amounts of input text …	\N	2019	https://arxiv.org/abs/1902.01718	t	42
2485	Reinforced cross-modal matching and self-supervised imitation learning for vision-language navigation	X Wang, Q Huang, A Celikyilmaz…	Vision-language navigation (VLN) is the task of navigating an embodied agent to carry out natural language instructions inside real 3D environments. In this paper, we study how to address three critical challenges for this task: the cross-modal grounding, the ill-posed …\n\nDoes not cite ELMo.	\N	2019	http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Reinforced_Cross-Modal_Matching_and_Self-Supervised_Imitation_Learning_for_Vision-Language_Navigation_CVPR_2019_paper.html	f	40
639	Deep learning-based document modeling for personality detection from text	N Majumder, S Poria, A Gelbukh…	This article presents a deep learning based method for determining the author's personality type from text: given a text, the presence or absence of the Big Five traits is detected in the author's psychological profile. For each of the five traits, the authors train a separate binary …	\N	2017	https://ieeexplore.ieee.org/abstract/document/7887639/	t	176
651	An empirical study and analysis of generalized zero-shot learning for object recognition in the wild	WL Chao, S Changpinyo, B Gong, F Sha	We investigate the problem of generalized zero-shot learning (GZSL). GZSL relaxes the unrealistic assumption in conventional zero-shot learning (ZSL) that test data belong only to unseen novel classes. In GZSL, test data might also come from seen classes and the …	\N	2016	https://link.springer.com/chapter/10.1007/978-3-319-46475-6_4	t	175
1289	Evaluation of sentence embeddings in downstream and linguistic probing tasks	CS Perone, R Silveira, TS Paula	Despite the fast developmental pace of new sentence embedding methods, it is still challenging to find comprehensive evaluations of these different techniques. In the past years, we saw significant improvements in the field of sentence embeddings and especially …	\N	2018	https://arxiv.org/abs/1806.06259	t	40
2581	Video-to-video synthesis	TC Wang, MY Liu, JY Zhu, G Liu, A Tao, J Kautz…	We study the problem of video-to-video synthesis, whose goal is to learn a mapping function from an input source video (eg, a sequence of semantic segmentation masks) to an output photorealistic video that precisely depicts the content of the source video. While its image …	\N	2018	https://arxiv.org/abs/1808.06601	t	174
634	Relation classification via multi-level attention cnns	L Wang, Z Cao, G De Melo, Z Liu	Relation classification is a crucial ingredient in numerous information extraction systems seeking to mine structured facts from text. We propose a novel convolutional neural network architecture for this task, relying on two levels of attention in order to better discern patterns …	\N	2016	https://www.aclweb.org/anthology/P16-1123.pdf	t	173
667	Fine-grained analysis of sentence embeddings using auxiliary prediction tasks	Y Adi, E Kermany, Y Belinkov, O Lavi…	There is a lot of research interest in encoding variable length sentences into fixed length vectors, in a way that preserves the sentence meanings. Two common methods include representations based on averaging word vectors, and representations based on the hidden …	\N	2016	https://arxiv.org/abs/1608.04207	t	170
621	Fine-grained opinion mining with recurrent neural networks and word embeddings	P Liu, S Joty, H Meng	The tasks in fine-grained opinion mining can be regarded as either a token-level sequence labeling problem or as a semantic compositional task. We propose a general class of discriminative models based on recurrent neural networks (RNNs) and word embeddings …	\N	2015	https://www.aclweb.org/anthology/D15-1168	t	169
660	Support vector machines and word2vec for text classification with semantic features	J Lilleberg, Y Zhu, Y Zhang	With the rapid expansion of new available information presented to us online on a daily basis, text classification becomes imperative in order to classify and maintain it. Word2vec offers a unique perspective to the text mining community. By converting words and phrases …	\N	2015	https://ieeexplore.ieee.org/abstract/document/7259377/	t	169
610	Political ideology detection using recursive neural networks	M Iyyer, P Enns, J Boyd-Graber, P Resnik	An individual's words often reveal their political ideology. Existing automated techniques to identify ideology from text focus on bags of words or wordlists, ignoring syntax. Taking inspiration from recent work in sentiment analysis that successfully models the …	\N	2014	https://www.aclweb.org/anthology/P14-1105	t	165
597	Transductive multi-view embedding for zero-shot recognition and annotation	Y Fu, TM Hospedales, T Xiang, Z Fu, S Gong	Most existing zero-shot learning approaches exploit transfer learning via an intermediate-level semantic representation such as visual attributes or semantic word vectors. Such a semantic representation is shared between an annotated auxiliary dataset and a target …	\N	2014	https://link.springer.com/chapter/10.1007/978-3-319-10605-2_38	t	164
644	Max-margin deepwalk: Discriminative learning of network representation.	C Tu, W Zhang, Z Liu, M Sun	DeepWalk is a typical representation learning method that learns low-dimensional representations for vertices in social networks. Similar to other network representation learning (NRL) models, it encodes the network structure into vertex representations and is …	\N	2016	http://weichengzhang.co/src/paper/ijcai2016_mmdw.pdf	t	164
1275	Syntax for semantic role labeling, to be, or not to be	S He, Z Li, H Zhao, H Bai	Semantic role labeling (SRL) is dedicated to recognizing the predicate-argument structure of a sentence. Previous studies have shown syntactic information has a remarkable contribution to SRL performance. However, such perception was challenged by a few recent …	\N	2018	https://www.aclweb.org/anthology/P18-1192.pdf	t	37
643	Massively multilingual word embeddings	W Ammar, G Mulcaire, Y Tsvetkov, G Lample…	We introduce new methods for estimating and evaluating embeddings of words in more than fifty languages in a single shared embedding space. Our estimation methods, multiCluster and multiCCA, use dictionaries and monolingual data; they do not require parallel data. Our …	\N	2016	https://arxiv.org/abs/1602.01925	t	162
1293	Can a suit of armor conduct electricity? a new dataset for open book question answering	T Mihaylov, P Clark, T Khot, A Sabharwal	We present a new kind of question answering dataset, OpenBookQA, modeled after open book exams for assessing human understanding of a subject. The open book that comes with our questions is a set of 1329 elementary level science facts. Roughly 6000 questions …	\N	2018	https://arxiv.org/abs/1809.02789	t	37
1312	Contextual augmentation: Data augmentation by words with paradigmatic relations	S Kobayashi	We propose a novel data augmentation for labeled sentences called contextual augmentation. We assume an invariance that sentences are natural even if the words in the sentences are replaced with other words with paradigmatic relations. We stochastically …	\N	2018	https://arxiv.org/abs/1805.06201	t	36
598	Zero-shot object recognition by semantic manifold distance	Z Fu, T Xiang, E Kodirov, S Gong	Object recognition by zero-shot learning (ZSL) aims to recognise objects without seeing any visual examples by learning knowledge transfer between seen and unseen object classes. This is typically achieved by exploring a semantic embedding space such as attribute space …	\N	2015	https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Fu_Zero-Shot_Object_Recognition_2015_CVPR_paper.html	t	160
626	Deep compositional captioning: Describing novel object categories without paired training data	L Anne Hendricks, S Venugopalan…	While recent deep neural network models have achieved promising results on the image captioning task, they rely largely on the availability of corpora with paired image and sentence captions to describe objects in context. In this work, we propose the Deep …	\N	2016	https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Hendricks_Deep_Compositional_Captioning_CVPR_2016_paper.html	t	158
646	Learning hierarchical representation model for nextbasket recommendation	P Wang, J Guo, Y Lan, J Xu, S Wan…	Next basket recommendation is a crucial task in market basket analysis. Given a user's purchase history, usually a sequence of transaction data, one attempts to build a recommender that can predict the next few items that the user most probably would like …	\N	2015	https://dl.acm.org/citation.cfm?id=2767694	t	158
1304	Read+ verify: Machine reading comprehension with unanswerable questions	M Hu, F Wei, Y Peng, Z Huang, N Yang…	Machine reading comprehension with unanswerable questions aims to abstain from answering when no answer can be inferred. Previous works using an additional no-answer option attempt to extract answers and detect unanswerable questions simultaneously, but …	\N	2019	https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4619	t	33
617	Associating neural word embeddings with deep image representations using fisher vectors	B Klein, G Lev, G Sadeh, L Wolf	In recent years, the problem of associating a sentence with an image has gained a lot of attention. This work continues to push the envelope and makes further progress in the performance of image annotation and image search by a sentence tasks. In this work, we …	\N	2015	https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Klein_Associating_Neural_Word_2015_CVPR_paper.html	t	157
1306	A comparison of techniques for language model integration in encoder-decoder speech recognition	S Toshniwal, A Kannan, CC Chiu, Y Wu…	Attention-based recurrent neural encoder-decoder models present an elegant solution to the automatic speech recognition problem. This approach folds the acoustic model, pronunciation model, and language model into a single network and requires only a parallel …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8639038/	t	32
688	Dimensional sentiment analysis using a regional CNN-LSTM model	J Wang, LC Yu, KR Lai, X Zhang	Dimensional sentiment analysis aims to recognize continuous numerical values in multiple dimensions such as the valencearousal (VA) space. Compared to the categorical approach that focuses on sentiment classification such as binary classification (ie, positive and …	\N	2016	https://www.aclweb.org/anthology/P16-2037.pdf	t	156
676	Neural sentiment classification with user and product attention	H Chen, M Sun, C Tu, Y Lin, Z Liu	Document-level sentiment classification aims to predict user's overall sentiment in a document about a product. However, most of existing methods only focus on local text information and ignore the global user preference and product characteristics. Even though …	\N	2016	https://www.aclweb.org/anthology/D16-1171	t	153
627	Convolutional neural network for paraphrase identification	W Yin, H Schütze	We present a new deep learning architecture Bi-CNN-MI for paraphrase identification (PI). Based on the insight that PI requires comparing two sentences on multiple levels of granularity, we learn multigranular sentence representations using convolutional neural …	\N	2015	https://www.aclweb.org/anthology/N15-1091	t	151
618	Rc-net: A general framework for incorporating knowledge into word representations	C Xu, Y Bai, J Bian, B Gao, G Wang, X Liu…	Representing words into vectors in continuous space can form up a potentially powerful basis to generate high-quality textual features for many text mining and natural language processing tasks. Some recent efforts, such as the skip-gram model, have attempted to learn …	\N	2014	https://dl.acm.org/citation.cfm?id=2662038	t	151
1322	Adversarial removal of demographic attributes from text data	Y Elazar, Y Goldberg	Recent advances in Representation Learning and Adversarial Training seem to succeed in removing unwanted features from the learned representation. We show that demographic information of authors is encoded in--and can be recovered from--the intermediate …	\N	2018	https://arxiv.org/abs/1808.06640	t	32
641	E-commerce in your inbox: Product recommendations at scale	M Grbovic, V Radosavljevic, N Djuric…	In recent years online advertising has become increasingly ubiquitous and effective. Advertisements shown to visitors fund sites and apps that publish digital content, manage social networks, and operate e-mail services. Given such large variety of internet resources …	\N	2015	https://dl.acm.org/citation.cfm?id=2788627	t	150
619	Learning image embeddings using convolutional neural networks for improved multi-modal semantics	D Kiela, L Bottou	We construct multi-modal concept representations by concatenating a skip-gram linguistic representation vector with a visual concept representation vector computed using the feature extraction layers of a deep convolutional neural network (CNN) trained on a large labeled …	\N	2014	https://www.aclweb.org/anthology/D14-1005	t	149
608	Building large-scale twitter-specific sentiment lexicon: A representation learning approach	D Tang, F Wei, B Qin, M Zhou, T Liu	In this paper, we propose to build large-scale sentiment lexicon from Twitter with a representation learning approach. We cast sentiment lexicon learning as a phrase-level sentiment classification task. The challenges are developing effective feature representation …	\N	2014	https://www.aclweb.org/anthology/C14-1018	t	148
2582	An end-to-end deep learning architecture for graph classification	M Zhang, Z Cui, M Neumann, Y Chen	Neural networks are typically designed to deal with data in tensor forms. In this paper, we propose a novel neural network architecture accepting graphs of arbitrary structure. Given a dataset containing graphs in the form of (G, y) where G is a graph and y is its class, we aim …	\N	2018	https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17146	t	148
1308	Neural cross-lingual named entity recognition with minimal resources	J Xie, Z Yang, G Neubig, NA Smith…	For languages with no annotated resources, unsupervised transfer of natural language processing models such as named-entity recognition (NER) from resource-rich languages would be an appealing capability. However, differences in words and word order across …	\N	2018	https://arxiv.org/abs/1808.09861	t	31
671	Revisiting visual question answering baselines	A Jabri, A Joulin, L Van Der Maaten	Visual question answering (VQA) is an interesting learning setting for evaluating the abilities and shortcomings of current systems for image understanding. Many of the recently proposed VQA systems include attention or memory mechanisms designed to perform …	\N	2016	https://link.springer.com/chapter/10.1007/978-3-319-46484-8_44	t	146
611	Fast and space-efficient entity linking for queries	R Blanco, G Ottaviano, E Meij	Entity linking deals with identifying entities from a knowledge base in a given piece of text and has become a fundamental building block for web search engines, enabling numerous downstream improvements from better document ranking to enhanced search results pages …	\N	2015	https://dl.acm.org/citation.cfm?id=2685317	t	145
653	A deeper look into sarcastic tweets using deep convolutional neural networks	S Poria, E Cambria, D Hazarika, P Vij	Sarcasm detection is a key task for many natural language processing tasks. In sentiment analysis, for example, sarcasm can flip the polarity of an" apparently positive" sentence and, hence, negatively affect polarity detection performance. To date, most approaches to …	\N	2016	https://arxiv.org/abs/1610.08815	t	143
702	Successor features for transfer in reinforcement learning	A Barreto, W Dabney, R Munos, JJ Hunt…	Transfer in reinforcement learning refers to the notion that generalization should occur not only within a task but also across tasks. We propose a transfer framework for the scenario where the reward function changes between tasks but the environment's dynamics remain …	\N	2017	http://papers.nips.cc/paper/6994-successor-features-for-transfer-in-reinforcement-learning	t	143
672	Multi-Domain Joint Semantic Frame Parsing Using Bi-Directional RNN-LSTM.	D Hakkani-Tür, G Tür, A Celikyilmaz…	Sequence-to-sequence deep learning has recently emerged as a new paradigm in supervised learning for spoken language understanding. However, most of the previous studies explored this framework for building single domain models for each task, such as …	\N	2016	https://pdfs.semanticscholar.org/d644/ae996755c803e067899bdd5ea52498d7091d.pdf	t	142
615	Learning a recurrent visual representation for image caption generation	X Chen, CL Zitnick	In this paper we explore the bi-directional mapping between images and their sentence-based descriptions. We propose learning this mapping using a recurrent neural network. Unlike previous approaches that map both sentences and images to a common embedding …	\N	2014	https://arxiv.org/abs/1411.5654	t	141
657	Temporal analysis of language through neural language models	Y Kim, YI Chiu, K Hanaki, D Hegde, S Petrov	We provide a method for automatically detecting change in language across time through a chronologically trained neural language model. We train the model on the Google Books Ngram corpus to obtain word vector representations specific to each year, and identify words …	\N	2014	https://arxiv.org/abs/1405.3515	t	141
690	Learning to respond with deep neural networks for retrieval-based human-computer conversation system	R Yan, Y Song, H Wu	To establish an automatic conversation system between humans and computers is regarded as one of the most hardcore problems in computer science, which involves interdisciplinary techniques in information retrieval, natural language processing, artificial intelligence, etc …	\N	2016	https://dl.acm.org/citation.cfm?id=2911542	t	140
637	Extractive summarization using continuous vector space models	M Kågebäck, O Mogren, N Tahmasebi…	Automatic summarization can help users extract the most important pieces of information from the vast amount of text digitized into electronic form everyday. Central to automatic summarization is the notion of similarity between sentences in text. In this paper we propose …	\N	2014	https://www.aclweb.org/anthology/W14-1504	t	138
666	Unifying distillation and privileged information	D Lopez-Paz, L Bottou, B Schölkopf…	Distillation (Hinton et al., 2015) and privileged information (Vapnik & Izmailov, 2015) are two techniques that enable machines to learn from other machines. This paper unifies these two techniques into generalized distillation, a framework to learn from multiple machines and …	\N	2015	https://arxiv.org/abs/1511.03643	t	137
1344	Transfertransfo: A transfer learning approach for neural network based conversational agents	T Wolf, V Sanh, J Chaumond, C Delangue	We introduce a new approach to generative data-driven dialogue systems (eg chatbots) called TransferTransfo which is a combination of a Transfer learning based training scheme and a high-capacity Transformer model. Fine-tuning is performed by using a multi-task …	\N	2019	https://arxiv.org/abs/1901.08149	t	31
1297	Modeling localness for self-attention networks	B Yang, Z Tu, DF Wong, F Meng, LS Chao…	Self-attention networks have proven to be of profound value for its strength of capturing global dependencies. In this work, we propose to model localness for self-attention networks, which enhances the ability of capturing useful local context. We cast localness …	\N	2018	https://arxiv.org/abs/1810.10182	t	30
1514	CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge	A Talmor, J Herzig, N Lourie, J Berant	When answering a question, people often draw upon their rich world knowledge in addition to some task-specific context. Recent work has focused primarily on answering questions based on some relevant document or content, and required very little general background …	\N	2018	https://arxiv.org/abs/1811.00937	t	30
1770	ATOMIC: An Atlas of Machine Commonsense for If-Then Reasoning	M Sap, R Le Bras, E Allaway…	We present ATOMIC, an atlas of everyday commonsense reasoning, organized through 300k textual descriptions. Compared to existing resources that center around taxonomic knowledge, ATOMIC focuses on inferential knowledge organized as typed if-then relations …	\N	2019	https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4160	t	29
658	Learning visual features from large weakly supervised data	A Joulin, L van der Maaten, A Jabri…	Convolutional networks trained on large supervised datasets produce visual features which form the basis for the state-of-the-art in many computer-vision problems. Further improvements of these visual features will likely require even larger manually labeled data …	\N	2016	https://link.springer.com/chapter/10.1007/978-3-319-46478-7_5	t	134
1286	Semi-supervised training for improving data efficiency in end-to-end speech synthesis	YA Chung, Y Wang, WN Hsu, Y Zhang…	Although end-to-end text-to-speech (TTS) models such as Tacotron have shown excellent results, they typically require a sizable set of high-quality< text, audio> pairs for training, which are expensive to collect. In this paper, we propose a semi-supervised training …	\N	2019	https://ieeexplore.ieee.org/abstract/document/8683862/	t	29
654	Siamese cbow: Optimizing word embeddings for sentence representations	T Kenter, A Borisov, M De Rijke	We present the Siamese Continuous Bag of Words (Siamese CBOW) model, a neural network for efficient estimation of high-quality sentence embeddings. Averaging the embeddings of words in a sentence has proven to be a surprisingly successful and efficient …	\N	2016	https://arxiv.org/abs/1606.04640	t	133
673	Emergent: a novel data-set for stance classification	W Ferreira, A Vlachos	We present Emergent, a novel data-set derived from a digital journalism project for rumour debunking. The data-set contains 300 rumoured claims and 2,595 associated news articles, collected and labelled by journalists with an estimation of their veracity (true, false or …	\N	2016	https://www.aclweb.org/anthology/N16-1138	t	133
668	: A Convolutional Net for Medical Records	P Nguyen, T Tran, N Wickramasinghe…	Feature engineering remains a major bottleneck when creating predictive systems from electronic medical records. At present, an important missing element is detecting predictive regular clinical motifs from irregular episodic records. We present Deepr (short for Deep …	\N	2016	https://ieeexplore.ieee.org/abstract/document/7762861/	t	133
674	Deepcare: A deep dynamic memory model for predictive medicine	T Pham, T Tran, D Phung, S Venkatesh	Personalized predictive medicine necessitates modeling of patient illness and care processes, which inherently have long-term temporal dependencies. Healthcare observations, recorded in electronic medical records, are episodic and irregular in time. We …	\N	2016	https://link.springer.com/chapter/10.1007/978-3-319-31750-2_3	t	132
2583	The history began from alexnet: A comprehensive survey on deep learning approaches	MZ Alom, TM Taha, C Yakopcic, S Westberg…	Deep learning has demonstrated tremendous success in variety of application domains in the past few years. This new field of machine learning has been growing rapidly and applied in most of the application domains with some new modalities of applications, which helps to …	\N	2018	https://arxiv.org/abs/1803.01164	t	132
2584	Representation learning with contrastive predictive coding	A Oord, Y Li, O Vinyals	While supervised learning has enabled great progress in many applications, unsupervised learning has not seen such widespread adoption, and remains an important and challenging endeavor for artificial intelligence. In this work, we propose a universal …	\N	2018	https://arxiv.org/abs/1807.03748	t	131
1468	SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems	A Wang, Y Pruksachatkun, N Nangia, A Singh…	In the last year, new models and methods for pretraining and transfer learning have driven striking performance improvements across a range of language understanding tasks. The GLUE benchmark, introduced one year ago, offers a single-number metric that summarizes …	\N	2019	https://arxiv.org/abs/1905.00537	t	29
635	Max-margin tensor neural network for Chinese word segmentation	W Pei, T Ge, B Chang	Recently, neural network models for natural language processing tasks have been increasingly focused on for their ability to alleviate the burden of manual feature engineering. In this paper, we propose a novel neural network model for Chinese word …	\N	2014	https://www.aclweb.org/anthology/P14-1028	t	129
661	A survey on the application of recurrent neural networks to statistical language modeling	W De Mulder, S Bethard, MF Moens	In this paper, we present a survey on the application of recurrent neural networks to the task of statistical language modeling. Although it has been shown that these models obtain good performance on this task, often superior to other state-of-the-art techniques, they suffer from …	\N	2015	https://www.sciencedirect.com/science/article/pii/S088523081400093X	t	129
684	From word embeddings to document similarities for improved information retrieval in software engineering	X Ye, H Shen, X Ma, R Bunescu, C Liu	The application of information retrieval techniques to search tasks in software engineering is made difficult by the lexical gap between search queries, usually expressed in natural language (eg English), and retrieved documents, usually expressed in code (eg …	\N	2016	https://dl.acm.org/citation.cfm?id=2884862	t	126
694	What to talk about and how? selective generation using lstms with coarse-to-fine alignment	H Mei, M Bansal, MR Walter	We propose an end-to-end, domain-independent neural encoder-aligner-decoder model for selective generation, ie, the joint task of content selection and surface realization. Our model first encodes a full set of over-determined database event records via an LSTM-based …	\N	2015	https://arxiv.org/abs/1509.00838	t	126
2585	Scaling egocentric vision: The epic-kitchens dataset	D Damen, H Doughty…	First-person vision is gaining interest as it offers a unique viewpoint on people's interaction with objects, their attention, and even intention. However, progress in this challenging domain has been relatively slow due to the lack of sufficiently large datasets. In this paper …	\N	2018	http://openaccess.thecvf.com/content_ECCV_2018/html/Dima_Damen_Scaling_Egocentric_Vision_ECCV_2018_paper.html	t	126
648	An analysis of the user occupational class through Twitter content	D Preoţiuc-Pietro, V Lampos, N Aletras	Social media content can be used as a complementary source to the traditional methods for extracting and studying collective social attributes. This study focuses on the prediction of the occupational class for a public user profile. Our analysis is conducted on a new …	\N	2015	https://www.aclweb.org/anthology/P15-1169	t	124
2586	Mean birds: Detecting aggression and bullying on twitter	D Chatzakou, N Kourtellis, J Blackburn…	In recent years, bullying and aggression against social media users have grown significantly, causing serious consequences to victims of all demographics. Nowadays, cyberbullying affects more than half of young social media users worldwide, suffering from …	\N	2017	https://dl.acm.org/citation.cfm?id=3091487	t	124
645	Multilingual distributed representations without word alignment	KM Hermann, P Blunsom	Distributed representations of meaning are a natural way to encode covariance relationships between words and phrases in NLP. By overcoming data sparsity problems, as well as providing information about semantic relatedness which is not available in discrete …	\N	2013	https://arxiv.org/abs/1312.6173	t	123
685	Continuous online sequence learning with an unsupervised neural network model	Y Cui, S Ahmad, J Hawkins	The ability to recognize and predict temporal sequences of sensory inputs is vital for survival in natural environments. Based on many known properties of cortical neurons, hierarchical temporal memory (HTM) sequence memory recently has been proposed as a theoretical …	\N	2016	https://www.mitpressjournals.org/doi/abs/10.1162/NECO_a_00893	t	123
686	Stance detection with bidirectional conditional encoding	I Augenstein, T Rocktäschel, A Vlachos…	Stance detection is the task of classifying the attitude expressed in a text towards a target such as Hillary Clinton to be" positive", negative" or" neutral". Previous work has assumed that either the target is mentioned in the text or that training data for every target is given …	\N	2016	https://arxiv.org/abs/1606.05464	t	122
2587	Next-generation machine learning for biological networks	DM Camacho, KM Collins, RK Powers, JC Costello…	Machine learning, a collection of data-analytical techniques aimed at building predictive models from multi-dimensional datasets, is becoming integral to modern biological research. By enabling one to generate models that learn from large datasets and make predictions on …	\N	2018	https://www.sciencedirect.com/science/article/pii/S0092867418305920	t	122
681	Cross-lingual models of word embeddings: An empirical comparison	S Upadhyay, M Faruqui, C Dyer, D Roth	Despite interest in using cross-lingual knowledge to learn word embeddings for various tasks, a systematic comparison of the possible approaches is lacking in the literature. We perform an extensive evaluation of four popular approaches of inducing cross-lingual …	\N	2016	https://arxiv.org/abs/1604.00425	t	120
2588	Summarizing source code using a neural attention model	S Iyer, I Konstas, A Cheung, L Zettlemoyer	High quality source code is often paired with high level summaries of the computation it performs, for example in code documentation or in descriptions posted in online forums. Such summaries are extremely useful for applications such as code search but are …	\N	2016	https://www.aclweb.org/anthology/P16-1195	t	119
2589	Joint event extraction via recurrent neural networks	TH Nguyen, K Cho, R Grishman	Event extraction is a particularly challenging problem in information extraction. The stateof-the-art models for this problem have either applied convolutional neural networks in a pipelined framework (Chen et al., 2015) or followed the joint architecture via structured …	\N	2016	https://www.aclweb.org/anthology/N16-1034	t	118
1527	Pooled Contextualized Embeddings for Named Entity Recognition	A Akbik, T Bergmann, R Vollgraf	Contextual string embeddings are a recent type of contextualized word embedding that were shown to yield state-of-the-art results when utilized in a range of sequence labeling tasks. They are based on character-level language models which treat text as distributions over …	\N	2019	https://www.aclweb.org/anthology/N19-1078.pdf	t	27
649	Evaluating word representation features in biomedical named entity recognition tasks	B Tang, H Cao, X Wang, Q Chen, H Xu	Biomedical Named Entity Recognition (BNER), which extracts important entities such as genes and proteins, is a crucial step of natural language processing in the biomedical domain. Various machine learning-based approaches have been applied to BNER tasks …	\N	2014	https://www.hindawi.com/journals/bmri/2014/240403/abs/	t	117
703	Improving document ranking with dual word embeddings	E Nalisnick, B Mitra, N Craswell…	This paper investigates the popular neural word embedding method Word2vec as a source of evidence in document ranking. In contrast to NLP applications of word2vec, which tend to use only the input embeddings, we retain both the input and the output embeddings …	\N	2016	https://dl.acm.org/citation.cfm?id=2889361	t	117
665	Knowledge-powered deep learning for word embedding	J Bian, B Gao, TY Liu	The basis of applying deep learning to solve natural language processing tasks is to obtain high-quality distributed representations of words, ie, word embeddings, from large amounts of text data. However, text itself usually contains incomplete and ambiguous information …	\N	2014	https://link.springer.com/chapter/10.1007/978-3-662-44848-9_9	t	116
1376	Sentence encoders on stilts: Supplementary training on intermediate labeled-data tasks	J Phang, T Févry, SR Bowman	Pretraining with language modeling and related unsupervised tasks has recently been shown to be a very effective enabling technology for the development of neural network models for language understanding tasks. In this work, we show that although language …	\N	2018	https://arxiv.org/abs/1811.01088	t	27
682	# tagspace: Semantic embeddings from hashtags	J Weston, S Chopra, K Adams	We describe a convolutional neural network that learns feature representations for short textual posts using hashtags as a supervised signal. The proposed approach is trained on up to 5.5 billion words predicting 100,000 possible hashtags. As well as strong performance …	\N	2014	https://www.aclweb.org/anthology/D14-1194	t	114
696	Sentence similarity learning by lexical decomposition and composition	Z Wang, H Mi, A Ittycheriah	Most conventional sentence similarity methods only focus on similar parts of two input sentences, and simply ignore the dissimilar parts, which usually give us some clues and semantic meanings about the sentences. In this work, we propose a model to take into …	\N	2016	https://arxiv.org/abs/1602.07019	t	114
713	Explaining human performance in psycholinguistic tasks with models of semantic similarity based on prediction and counting: A review and empirical validation	P Mandera, E Keuleers, M Brysbaert	Recent developments in distributional semantics (Mikolov, Chen, Corrado, & Dean, 2013; Mikolov, Sutskever, Chen, Corrado, & Dean, 2013) include a new class of prediction-based models that are trained on a text corpus and that measure semantic similarity between …	\N	2017	https://www.sciencedirect.com/science/article/pii/S0749596X16300079	t	114
2590	Connecting social media to e-commerce: Cold-start product recommendation using microblogging information	WX Zhao, S Li, Y He, EY Chang…	In recent years, the boundaries between e-commerce and social networking have become increasingly blurred. Many e-commerce Web sites support the mechanism of social login where users can sign on the Web sites using their social network identities such as their …	\N	2015	https://ieeexplore.ieee.org/abstract/document/7355341/	t	113
693	Word embeddings for speech recognition	S Bengio, G Heigold	Speech recognition systems have used the concept of states as a way to decompose words into sub-word units for decades. As the number of such states now reaches the number of words used to train acoustic models, it is interesting to consider approaches that relax the …	\N	2014	https://www.isca-speech.org/archive/interspeech_2014/i14_1053.html	t	112
663	Revisiting embedding features for simple semi-supervised learning	J Guo, W Che, H Wang, T Liu	Recent work has shown success in using continuous word embeddings learned from unlabeled data as features to improve supervised NLP systems, which is regarded as a simple semi-supervised learning mechanism. However, fundamental problems on effectively …	\N	2014	https://www.aclweb.org/anthology/D14-1012	t	111
679	A probabilistic model for learning multi-prototype word embeddings	F Tian, H Dai, J Bian, B Gao, R Zhang, E Chen…	Distributed word representations have been widely used and proven to be useful in quite a few natural language processing and text mining tasks. Most of existing word embedding models aim at generating only one embedding vector for each individual word, which …	\N	2014	https://www.aclweb.org/anthology/C14-1016	t	111
695	Evaluating prerequisite qualities for learning end-to-end dialog systems	J Dodge, A Gane, X Zhang, A Bordes, S Chopra…	A long-term goal of machine learning is to build intelligent conversational agents. One recent popular approach is to train end-to-end models on a large amount of real dialog transcripts between humans (Sordoni et al., 2015; Vinyals & Le, 2015; Shang et al., 2015) …	\N	2015	https://arxiv.org/abs/1511.06931	t	111
2591	Ridge regression, hubness, and zero-shot learning	Y Shigeto, I Suzuki, K Hara, M Shimbo…	This paper discusses the effect of hubness in zero-shot learning, when ridge regression is used to find a mapping between the example space to the label space. Contrary to the existing approach, which attempts to find a mapping from the example space to the label …	\N	2015	https://link.springer.com/chapter/10.1007/978-3-319-23528-8_9	t	110
2592	Network representation learning: A survey	D Zhang, J Yin, X Zhu, C Zhang	With the widespread use of information technologies, information networks are becoming increasingly popular to capture complex relationships across various disciplines. In reality, the large scale of information networks often makes network analytic tasks computationally …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8395024/	t	109
2593	Efficient softmax approximation for GPUs	E Grave, A Joulin, M Cissé, H Jégou	We propose an approximate strategy to efficiently train neural network based language models over very large vocabularies. Our approach, called adaptive softmax, circumvents the linear dependency on the vocabulary size by exploiting the unbalanced word distribution …	\N	2017	https://dl.acm.org/citation.cfm?id=3305516	t	108
2594	Starspace: Embed all the things!	LY Wu, A Fisch, S Chopra, K Adams, A Bordes…	We present StarSpace, a general-purpose neural embedding model that can solve a wide variety of problems: labeling tasks such as text classification, ranking tasks such as information retrieval/web search, collaborative filtering-based or content-based …	\N	2018	https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16998	t	108
2595	Distant supervision for relation extraction with sentence-level attention and entity descriptions	G Ji, K Liu, S He, J Zhao	Distant supervision for relation extraction is an efficient method to scale relation extraction to very large corpora which contains thousands of relations. However, the existing approaches have flaws on selecting valid instances and lack of background knowledge about the …	\N	2017	https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14491	t	108
2596	Gated neural networks for targeted sentiment analysis	M Zhang, Y Zhang, DT Vo	Targeted sentiment analysis classifies the sentiment polarity towards each target entity mention in given text documents. Seminal methods extract manual discrete features from automatic syntactic parse trees in order to capture semantic information of the enclosing …	\N	2016	https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12074	t	107
704	Show me your evidence-an automatic method for context dependent evidence detection	R Rinott, L Dankin, CA Perez, MM Khapra…	Engaging in a debate with oneself or others to take decisions is an integral part of our day-today life. A debate on a topic (say, use of performance enhancing drugs) typically proceeds by one party making an assertion/claim (say, PEDs are bad for health) and then providing an …	\N	2015	https://www.aclweb.org/anthology/D15-1050	t	106
2597	Tweet2vec: Character-based distributed representations for social media	B Dhingra, Z Zhou, D Fitzpatrick, M Muehl…	Text from social media provides a set of challenges that can cause traditional NLP approaches to fail. Informal language, spelling errors, abbreviations, and special characters are all commonplace in these posts, leading to a prohibitively large vocabulary size for word …	\N	2016	https://arxiv.org/abs/1605.03481	t	106
2598	Predicting multicellular function through multi-layer tissue networks	M Zitnik, J Leskovec	Motivation Understanding functions of proteins in specific human tissues is essential for insights into disease diagnostics and therapeutics, yet prediction of tissue-specific cellular function remains a critical challenge for biomedicine. Results Here, we present OhmNet, a …	\N	2017	https://academic.oup.com/bioinformatics/article-abstract/33/14/i190/3953967	t	106
2599	Neural word segmentation learning for Chinese	D Cai, H Zhao	Most previous approaches to Chinese word segmentation formalize this problem as a character-based sequence labeling task where only contextual information within fixed sized local windows and simple interactions between adjacent tags can be captured. In this paper …	\N	2016	https://arxiv.org/abs/1606.04300	t	105
2600	Nasari: Integrating explicit knowledge and corpus statistics for a multilingual representation of concepts and entities	J Camacho-Collados, MT Pilehvar, R Navigli	Owing to the need for a deep understanding of linguistic items, semantic representation is considered to be one of the fundamental components of several applications in Natural Language Processing and Artificial Intelligence. As a result, semantic representation has …	\N	2016	https://www.sciencedirect.com/science/article/pii/S0004370216300820	t	105
2601	Deep learning techniques for music generation-a survey	JP Briot, G Hadjeres, F Pachet	This book is a survey and an analysis of different ways of using deep learning (deep artificial neural networks) to generate musical content. At first, we propose a methodology based on four dimensions for our analysis:-objective-What musical content is to be generated?(eg …	\N	2017	https://arxiv.org/abs/1709.01620	t	105
699	A bayesian mixed effects model of literary character	D Bamman, T Underwood, NA Smith	We consider the problem of automatically inferring latent character types in a collection of 15,099 English novels published between 1700 and 1899. Unlike prior work in which character types are assumed responsible for probabilistically generating all text associated …	\N	2014	https://www.aclweb.org/anthology/P14-1035	t	104
689	Hammering towards QED	JC Blanchette, C Kaliszyk, LC Paulson…	The main ingredients underlying this approach are efficient automatic theorem provers that can cope with hundreds of axioms, suitable translations of the proof assistant's logic to the logic of the automatic provers, heuristic and learning methods that select relevant facts from …	\N	2016	https://pure.mpg.de/rest/items/item_2381986/component/file_2381985/content	t	104
2602	Medical semantic similarity with a neural language model	L De Vine, G Zuccon, B Koopman, L Sitbon…	Advances in neural network language models have demonstrated that these models can effectively learn representations of words meaning. In this paper, we explore a variation of neural language models that can learn on concepts taken from structured ontologies and …	\N	2014	https://dl.acm.org/citation.cfm?id=2661974	t	102
725	Automatic image annotation using deep learning representations	VN Murthy, S Maji, R Manmatha	We propose simple and effective models for the image annotation that make use of Convolutional Neural Network (CNN) features extracted from an image and word embedding vectors to represent their associated tags. Our first set of models is based on the …	\N	2015	https://dl.acm.org/citation.cfm?id=2749391	t	101
2603	Do supervised distributional methods really learn lexical inference relations?	O Levy, S Remus, C Biemann, I Dagan	Distributional representations of words have been recently used in supervised settings for recognizing lexical inference relations between word pairs, such as hypernymy and entailment. We investigate a collection of these state-of-the-art methods, and show that they …	\N	2015	https://www.aclweb.org/anthology/N15-1098	t	101
1274	Contextualized word representations for reading comprehension	S Salant, J Berant	Reading a document and extracting an answer to a question about its content has attracted substantial attention recently. While most work has focused on the interaction between the question and the document, in this work we evaluate the importance of context when the …	\N	2017	https://arxiv.org/abs/1712.03609	t	26
2604	Analyzing the behavior of visual question answering models	A Agrawal, D Batra, D Parikh	Recently, a number of deep-learning based models have been proposed for the task of Visual Question Answering (VQA). The performance of most models is clustered around 60-70%. In this paper we propose systematic methods to analyze the behavior of these models …	\N	2016	https://arxiv.org/abs/1606.07356	t	100
2605	Representation learning for very short texts using weighted word embedding aggregation	C De Boom, S Van Canneyt, T Demeester…	Short text messages such as tweets are very noisy and sparse in their use of vocabulary. Traditional textual representations, such as tf-idf, have difficulty grasping the semantic meaning of such texts, which is important in applications such as event detection, opinion …	\N	2016	https://www.sciencedirect.com/science/article/pii/S0167865516301362	t	99
2606	Task-guided and path-augmented heterogeneous network embedding for author identification	T Chen, Y Sun	In this paper, we study the problem of author identification under double-blind review setting, which is to identify potential authors given information of an anonymized paper. Different from existing approaches that rely heavily on feature engineering, we propose to use …	\N	2017	https://dl.acm.org/citation.cfm?id=3018735	t	99
2607	subgraph2vec: Learning distributed representations of rooted sub-graphs from large graphs	A Narayanan, M Chandramohan, L Chen, Y Liu…	In this paper, we present subgraph2vec, a novel approach for learning latent representations of rooted subgraphs from large graphs inspired by recent advancements in Deep Learning and Graph Kernels. These latent representations encode semantic …	\N	2016	https://arxiv.org/abs/1606.08928	t	99
2608	Dynamic word embeddings	R Bamler, S Mandt	We present a probabilistic language model for time-stamped text data which tracks the semantic evolution of individual words over time. The model represents words and contexts by latent trajectories in an embedding space. At each moment in time, the embedding …	\N	2017	https://dl.acm.org/citation.cfm?id=3305421	t	98
2609	Is neural machine translation ready for deployment? A case study on 30 translation directions	M Junczys-Dowmunt, T Dwojak, H Hoang	In this paper we provide the largest published comparison of translation quality for phrase-based SMT and neural machine translation across 30 translation directions. For ten directions we also include hierarchical phrase-based MT. Experiments are performed for the …	\N	2016	https://arxiv.org/abs/1610.01108	t	97
1281	NTUA-SLP at SemEval-2018 Task 1: predicting affective content in tweets with deep attentive RNNs and transfer learning	C Baziotis, N Athanasiou, A Chronopoulou…	In this paper we present deep-learning models that submitted to the SemEval-2018 Task~ 1 competition:" Affect in Tweets". We participated in all subtasks for English tweets. We propose a Bi-LSTM architecture equipped with a multi-layer self attention mechanism. The …	\N	2018	https://arxiv.org/abs/1804.06658	t	26
1311	Deep relevance ranking using enhanced document-query interactions	R McDonald, GI Brokos, I Androutsopoulos	We explore several new models for document relevance ranking, building upon the Deep Relevance Matching Model (DRMM) of Guo et al.(2016). Unlike DRMM, which uses context-insensitive encodings of terms and query-document term interactions, we inject rich context …	\N	2018	https://arxiv.org/abs/1809.01682	t	26
2610	Visual relationship detection with internal and external linguistic knowledge distillation	R Yu, A Li, VI Morariu, LS Davis	Understanding the visual relationship between two objects involves identifying the subject, the object, and a predicate relating them. We leverage the strong correlations between the predicate and the (subj, obj) pair (both semantically and spatially) to predict predicates …	\N	2017	http://openaccess.thecvf.com/content_iccv_2017/html/Yu_Visual_Relationship_Detection_ICCV_2017_paper.html	t	96
2611	Zero-shot recognition via semantic embeddings and knowledge graphs	X Wang, Y Ye, A Gupta	We consider the problem of zero-shot recognition: learning a visual classifier for a category with zero training examples, just using the word embedding of the category and its relationship to other categories, which visual data are provided. The key to dealing with the …	\N	2018	http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Zero-Shot_Recognition_via_CVPR_2018_paper.html	t	96
1305	Exploring graph-structured passage representation for multi-hop reading comprehension with graph neural networks	L Song, Z Wang, M Yu, Y Zhang, R Florian…	Multi-hop reading comprehension focuses on one type of factoid question, where a system needs to properly integrate multiple pieces of evidence to correctly answer a question. Previous work approximates global evidence with local coreference information, encoding …	\N	2018	https://arxiv.org/abs/1809.02040	t	24
2612	Breaking sticks and ambiguities with adaptive skip-gram	S Bartunov, D Kondrashkin, A Osokin…	The recently proposed Skip-gram model is a powerful method for learning high-dimensional word representations that capture rich semantic relationships between words. However, Skipgram as well as most prior work on learning word representations does not take into …	\N	2016	http://www.jmlr.org/proceedings/papers/v51/bartunov16.pdf	t	94
2613	Transfer learning for music classification and regression tasks	K Choi, G Fazekas, M Sandler, K Cho	In this paper, we present a transfer learning approach for music classification and regression tasks. We propose to use a pre-trained convnet feature, a concatenated feature vector using the activations of feature maps of multiple layers in a trained convolutional network. We …	\N	2017	https://arxiv.org/abs/1703.09179	t	94
2614	Query expansion using word embeddings	S Kuzi, A Shtok, O Kurland	We present a suite of query expansion methods that are based on word embeddings. Using Word2Vec's CBOW embedding approach, applied over the entire corpus on which search is performed, we select terms that are semantically related to the query. Our methods either …	\N	2016	https://dl.acm.org/citation.cfm?id=2983876	t	94
2615	Recursive neural conditional random fields for aspect-based sentiment analysis	W Wang, SJ Pan, D Dahlmeier, X Xiao	In aspect-based sentiment analysis, extracting aspect terms along with the opinions being expressed from user-generated content is one of the most important subtasks. Previous studies have shown that exploiting connections between aspect and opinion terms is …	\N	2016	https://arxiv.org/abs/1603.06679	t	94
2616	Optimizing the latent space of generative networks	P Bojanowski, A Joulin, D Lopez-Paz…	Generative Adversarial Networks (GANs) have been shown to be able to sample impressively realistic images. GAN training consists of a saddle point optimization problem that can be thought of as an adversarial game between a generator which produces the …	\N	2017	https://arxiv.org/abs/1707.05776	t	94
2617	Neural CRF parsing	G Durrett, D Klein	This paper describes a parsing model that combines the exact dynamic programming of CRF parsing with the rich nonlinear featurization of neural net approaches. Our model is structurally a CRF that factors over anchored rule productions, but instead of linear potential …	\N	2015	https://arxiv.org/abs/1507.03641	t	93
2618	Twitter as a lifeline: Human-annotated twitter corpora for NLP of crisis-related messages	M Imran, P Mitra, C Castillo	Microblogging platforms such as Twitter provide active communication channels during mass convergence and emergency events such as earthquakes, typhoons. During the sudden onset of a crisis situation, affected people post useful information on Twitter that can …	\N	2016	https://arxiv.org/abs/1605.05894	t	93
2619	Word sense disambiguation: A unified evaluation framework and empirical comparison	A Raganato, J Camacho-Collados…	Word Sense Disambiguation is a long-standing task in Natural Language Processing, lying at the core of human language understanding. However, the evaluation of automatic systems has been problematic, mainly due to the lack of a reliable evaluation …	\N	2017	https://www.aclweb.org/anthology/papers/E/E17/E17-1010/	t	93
2620	Memory aware synapses: Learning what (not) to forget	R Aljundi, F Babiloni, M Elhoseiny…	Humans can learn in a continuous manner. Old rarely utilized knowledge can be overwritten by new incoming information while important, frequently used knowledge is prevented from being erased. In artificial learning systems, lifelong learning so far has focused mainly on …	\N	2018	http://openaccess.thecvf.com/content_ECCV_2018/html/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.html	t	93
2621	Deep convolutional acoustic word embeddings using word-pair side information	H Kamper, W Wang, K Livescu	Recent studies have been revisiting whole words as the basic modelling unit in speech recognition and query applications, instead of phonetic units. Such whole-word segmental systems rely on a function that maps a variable-length speech segment to a vector in a fixed …	\N	2016	https://ieeexplore.ieee.org/abstract/document/7472619/	t	92
2622	Sensing spatial distribution of urban land use by integrating points-of-interest and Google Word2Vec model	Y Yao, X Li, X Liu, P Liu, Z Liang…	Urban land use information plays an essential role in a wide variety of urban planning and environmental monitoring processes. During the past few decades, with the rapid technological development of remote sensing (RS), geographic information systems (GIS) …	\N	2017	https://www.tandfonline.com/doi/abs/10.1080/13658816.2016.1244608	t	92
692	Low-rank tensors for scoring dependency structures	T Lei, Y Xin, Y Zhang, R Barzilay…	Accurate scoring of syntactic structures such as head-modifier arcs in dependency parsing typically requires rich, highdimensional feature representations. A small subset of such features is often selected manually. This is problematic when features lack clear linguistic …	\N	2014	https://www.aclweb.org/anthology/P14-1130	t	91
1303	A qualitative comparison of coqa, squad 2.0 and quac	M Yatskar	In this work, we compare three datasets which build on the paradigm defined in SQuAD for question answering: SQuAD 2.0, QuAC, and CoQA. We compare these three datasets along several of their new features:(1) unanswerable questions,(2) multi-turn interactions, and (3) …	\N	2018	https://arxiv.org/abs/1809.10735	t	23
2623	Unsupervised learning by predicting noise	P Bojanowski, A Joulin	Convolutional neural networks provide visual features that perform well in many computer vision applications. However, training these networks requires large amounts of supervision; this paper introduces a generic framework to train such networks, end-to-end, with no …	\N	2017	https://dl.acm.org/citation.cfm?id=3305435	t	91
2624	Attending to characters in neural sequence labeling models	M Rei, GKO Crichton, S Pyysalo	Sequence labeling architectures use word embeddings for capturing similarity, but suffer when handling previously unseen or rare words. We investigate character-level extensions to such models and propose a novel architecture for combining alternative word …	\N	2016	https://arxiv.org/abs/1611.04361	t	90
2625	Objects2action: Classifying and localizing actions without any video example	M Jain, JC van Gemert, T Mensink…	The goal of this paper is to recognize actions in video without the need for examples. Different from traditional zero-shot approaches we do not demand the design and specification of attribute classifiers and class-to-attribute mappings to allow for transfer from …	\N	2015	http://openaccess.thecvf.com/content_iccv_2015/html/Jain_Objects2action_Classifying_and_ICCV_2015_paper.html	t	89
2626	Attention with intention for a neural network conversation model	K Yao, G Zweig, B Peng	In a conversation or a dialogue process, attention and intention play intrinsic roles. This paper proposes a neural network based approach that models the attention and intention processes. It essentially consists of three recurrent networks. The encoder network is a word …	\N	2015	https://arxiv.org/abs/1510.08565	t	89
2627	Signed network embedding in social media	S Wang, J Tang, C Aggarwal, Y Chang, H Liu	Network embedding is to learn low-dimensional vector representations for nodes of a given social network, facilitating many tasks in social network analysis such as link prediction. The vast majority of existing embedding algorithms are designed for unsigned social networks or …	\N	2017	https://epubs.siam.org/doi/abs/10.1137/1.9781611974973.37	t	89
1747	Commonsense for Generative Multi-Hop Question Answering Tasks	L Bauer, Y Wang, M Bansal	Reading comprehension QA tasks have seen a recent surge in popularity, yet most works have focused on fact-finding extractive QA. We instead focus on a more challenging multi-hop generative task (NarrativeQA), which requires the model to reason, gather, and …	\N	2018	https://arxiv.org/abs/1809.06309	t	22
1288	A Full End-to-End Semantic Role Labeler, Syntax-agnostic Over Syntax-aware?	J Cai, S He, Z Li, H Zhao	Semantic role labeling (SRL) is to recognize the predicate-argument structure of a sentence, including subtasks of predicate disambiguation and argument labeling. Previous studies usually formulate the entire SRL problem into two or more subtasks. For the first time, this …	\N	2018	https://arxiv.org/abs/1808.03815	t	21
2628	The science of science: From the perspective of complex systems	A Zeng, Z Shen, J Zhou, J Wu, Y Fan, Y Wang…	The science of science (SOS) is a rapidly developing field which aims to understand, quantify and predict scientific research and the resulting outcomes. The problem is essentially related to almost all scientific disciplines and thus has attracted attention of …	\N	2017	https://www.sciencedirect.com/science/article/pii/S0370157317303289	t	87
2629	Mining user opinions in mobile app reviews: A keyword-based approach (t)	PM Vu, TT Nguyen, HV Pham…	User reviews of mobile apps often contain complaints or suggestions which are valuable for app developers to improve user experience and satisfaction. However, due to the large volume and noisy-nature of those reviews, manually analyzing them for useful opinions is …	\N	2015	https://ieeexplore.ieee.org/abstract/document/7372063/	t	86
2630	Automatic text scoring using neural networks	D Alikaniotis, H Yannakoudakis, M Rei	Automated Text Scoring (ATS) provides a cost-effective and consistent alternative to human marking. However, in order to achieve good performance, the predictive features of the system need to be manually engineered by human experts. We introduce a model that forms …	\N	2016	https://arxiv.org/abs/1606.04289	t	86
2631	Poseidon: An efficient communication architecture for distributed deep learning on {GPU} clusters	H Zhang, Z Zheng, S Xu, W Dai, Q Ho, X Liang…	Deep learning models can take weeks to train on a single GPU-equipped machine, necessitating scaling out DL training to a GPU-cluster. However, current distributed DL implementations can scale poorly due to substantial parameter synchronization over the …	\N	2017	https://www.usenix.org/conference/atc17/technical-sessions/presentation/zhang	t	86
1316	Multiway Attention Networks for Modeling Sentence Pairs.	C Tan, F Wei, W Wang, W Lv, M Zhou	Modeling sentence pairs plays the vital role for judging the relationship between two sentences, such as paraphrase identification, natural language inference, and answer sentence selection. Previous work achieves very promising results using neural networks …	\N	2018	https://pdfs.semanticscholar.org/2b32/b4fa1e28c256745f1573b5444b1b2c8df30e.pdf	t	21
2632	Discourse complements lexical semantics for non-factoid answer reranking	P Jansen, M Surdeanu, P Clark	We propose a robust answer reranking model for non-factoid questions that integrates lexical semantics with discourse information, driven by two representations of discourse: a shallow representation centered around discourse markers, and a deep one based on …	\N	2014	https://www.aclweb.org/anthology/P14-1092	t	85
2633	Ensemble application of convolutional neural networks and multiple kernel learning for multimodal sentiment analysis	S Poria, H Peng, A Hussain, N Howard, E Cambria	The advent of the Social Web has enabled anyone with an Internet connection to easily create and share their ideas, opinions and content with millions of other people around the world. In pace with a global deluge of videos from billions of computers, smartphones …	\N	2017	https://www.sciencedirect.com/science/article/pii/S0925231217302023	t	85
2634	Phrase localization and visual relationship detection with comprehensive image-language cues	BA Plummer, A Mallya, CM Cervantes…	This paper presents a framework for localization or grounding of phrases in images using a large collection of linguistic and visual cues. We model the appearance, size, and position of entity bounding boxes, adjectives that contain attribute information, and spatial relationships …	\N	2017	http://openaccess.thecvf.com/content_iccv_2017/html/Plummer_Phrase_Localization_and_ICCV_2017_paper.html	t	85
2635	Toward large-scale vulnerability discovery using machine learning	G Grieco, GL Grinblat, L Uzal, S Rawat, J Feist…	With sustained growth of software complexity, finding security vulnerabilities in operating systems has become an important necessity. Nowadays, OS are shipped with thousands of binary executables. Unfortunately, methodologies and tools for an OS scale program testing …	\N	2016	https://dl.acm.org/citation.cfm?id=2857720	t	85
2636	Visual madlibs: Fill in the blank description generation and question answering	L Yu, E Park, AC Berg, TL Berg	In this paper, we introduce a new dataset consisting of 360,001 focused natural language descriptions for 10,738 images. This dataset, the Visual Madlibs dataset, is collected using automatically produced fill-in-the-blank templates designed to gather targeted descriptions …	\N	2015	http://openaccess.thecvf.com/content_iccv_2015/html/Yu_Visual_Madlibs_Fill_ICCV_2015_paper.html	t	84
2637	The meaning factory: Formal semantics for recognizing textual entailment and determining semantic similarity	J Bjerva, J Bos, R Van der Goot, M Nissim	Shared Task 1 of SemEval-2014 comprised two subtasks on the same dataset of sentence pairs: recognizing textual entailment and determining textual similarity. We used an existing system based on formal semantics and logical inference to participate in the first …	\N	2014	https://www.aclweb.org/anthology/S14-2114	t	83
2638	Combining recurrent and convolutional neural networks for relation classification	NT Vu, H Adel, P Gupta, H Schütze	This paper investigates two different neural architectures for the task of relation classification: convolutional neural networks and recurrent neural networks. For both models, we demonstrate the effect of different architectural choices. We present a new …	\N	2016	https://arxiv.org/abs/1605.07333	t	82
2639	Effective deep learning-based multi-modal retrieval	W Wang, X Yang, BC Ooi, D Zhang…	Multi-modal retrieval is emerging as a new search paradigm that enables seamless information retrieval from various types of media. For example, users can simply snap a movie poster to search for relevant reviews and trailers. The mainstream solution to the …	\N	2016	https://dl.acm.org/citation.cfm?id=2884421	t	81
2640	Learning to reweight terms with distributed representations	G Zheng, J Callan	Term weighting is a fundamental problem in IR research and numerous weighting models have been proposed. Proper term weighting can greatly improve retrieval accuracies, which essentially involves two types of query understanding: interpreting the query and judging the …	\N	2015	https://dl.acm.org/citation.cfm?id=2767700	t	81
2641	Adversarial multi-criteria learning for chinese word segmentation	X Chen, Z Shi, X Qiu, X Huang	Different linguistic perspectives causes many diverse segmentation criteria for Chinese word segmentation (CWS). Most existing methods focus on improve the performance for each single criterion. However, it is interesting to exploit these different criteria and mining their …	\N	2017	https://arxiv.org/abs/1704.07556	t	81
2642	Using word embeddings in twitter election classification	X Yang, C Macdonald, I Ounis	Word embeddings and convolutional neural networks (CNN) have attracted extensive attention in various classification tasks for Twitter, eg sentiment classification. However, the effect of the configuration used to generate the word embeddings on the classification …	\N	2018	https://link.springer.com/article/10.1007/s10791-017-9319-5	t	81
2643	Symbol emergence in robotics: a survey	T Taniguchi, T Nagai, T Nakamura, N Iwahashi…	Humans can learn a language through physical interaction with their environment and semiotic communication with other people. It is very important to obtain a computational understanding of how humans can form symbol systems and obtain semiotic skills through …	\N	2016	https://www.tandfonline.com/doi/abs/10.1080/01691864.2016.1164622	t	80
2644	Semi-supervised word sense disambiguation using word embeddings in general and specific domains	K Taghipour, HT Ng	One of the weaknesses of current supervised word sense disambiguation (WSD) systems is that they only treat a word as a discrete entity. However, a continuous-space representation of words (word embeddings) can provide valuable information and thus improve …	\N	2015	https://www.aclweb.org/anthology/N15-1035	t	80
2645	Fvqa: Fact-based visual question answering	P Wang, Q Wu, C Shen, A Dick…	Visual Question Answering (VQA) has attracted much attention in both computer vision and natural language processing communities, not least because it offers insight into the relationships between two important sources of information. Current datasets, and the …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8046084/	t	80
2646	Context-and content-aware embeddings for query rewriting in sponsored search	M Grbovic, N Djuric, V Radosavljevic…	Search engines represent one of the most popular web services, visited by more than 85% of internet users on a daily basis. Advertisers are interested in making use of this vast business potential, as very clear intent signal communicated through the issued query …	\N	2015	https://dl.acm.org/citation.cfm?id=2767709	t	79
2647	Combining retrieval, statistics, and inference to answer elementary science questions	P Clark, O Etzioni, T Khot, A Sabharwal…	What capabilities are required for an AI system to pass standard 4th Grade Science Tests? Previous work has examined the use of Markov Logic Networks (MLNs) to represent the requisite background knowledge and interpret test questions, but did not improve upon an …	\N	2016	https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/11963	t	79
2648	Learning distributed representations from reviews for collaborative filtering	A Almahairi, K Kastner, K Cho, A Courville	Recent work has shown that collaborative filter-based recommender systems can be improved by incorporating side information, such as natural language reviews, as a way of regularizing the derived product representations. Motivated by the success of this approach …	\N	2015	https://dl.acm.org/citation.cfm?id=2800192	t	79
2649	A novel neural topic model and its supervised extension	Z Cao, S Li, Y Liu, W Li, H Ji	Topic modeling techniques have the benefits of modeling words and documents uniformly under a probabilistic framework. However, they also suffer from the limitations of sensitivity to initialization and unigram topic distribution, which can be remedied by deep learning …	\N	2015	https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9303	t	78
2650	Supervised word mover's distance	G Huang, C Guo, MJ Kusner, Y Sun, F Sha…	Accurately measuring the similarity between text documents lies at the core of many real world applications of machine learning. These include web-search ranking, document recommendation, multi-lingual document matching, and article categorization. Recently, a …	\N	2016	http://papers.nips.cc/paper/6138-supervised-word-movers-distance	t	78
2651	Hierarchical neural language models for joint representation of streaming documents and their content	N Djuric, H Wu, V Radosavljevic, M Grbovic…	We consider the problem of learning distributed representations for documents in data streams. The documents are represented as low-dimensional vectors and are jointly learned with distributed vector representations of word tokens using a hierarchical framework with …	\N	2015	https://dl.acm.org/citation.cfm?id=2741643	t	77
2652	Multi-timescale long short-term memory neural network for modelling sentences and documents	P Liu, X Qiu, X Chen, S Wu, X Huang	Neural network based methods have obtained great progress on a variety of natural language processing tasks. However, it is still a challenge task to model long texts, such as sentences and documents. In this paper, we propose a multi-timescale long short-term …	\N	2015	https://www.aclweb.org/anthology/D15-1280	t	77
2653	Learning a natural language interface with neural programmer	A Neelakantan, QV Le, M Abadi, A McCallum…	Learning a natural language interface for database tables is a challenging task that involves deep language understanding and multi-step reasoning. The task is often approached by mapping natural language queries to logical forms or programs that provide the desired …	\N	2016	https://arxiv.org/abs/1611.08945	t	77
2654	Analogy-based detection of morphological and semantic relations with word embeddings: what works and what doesn't.	A Gladkova, A Drozd, S Matsuoka	Following up on numerous reports of analogybased identification of “linguistic regularities” in word embeddings, this study applies the widely used vector offset method to 4 types of linguistic relations: inflectional and derivational morphology, and lexicographic and …	\N	2016	https://www.aclweb.org/anthology/N16-2002	t	76
2655	What's cookin'? interpreting cooking videos using text, speech and vision	J Malmaud, J Huang, V Rathod, N Johnston…	We present a novel method for aligning a sequence of instructions to a video of someone carrying out a task. In particular, we focus on the cooking domain, where the instructions correspond to the recipe. Our technique relies on an HMM to align the recipe steps to the …	\N	2015	https://arxiv.org/abs/1503.01558	t	75
2656	Gated recursive neural network for Chinese word segmentation	X Chen, X Qiu, C Zhu, X Huang	Recently, neural network models for natural language processing tasks have been increasingly focused on for their ability of alleviating the burden of manual feature engineering. However, the previous neural models cannot extract the complicated feature …	\N	2015	https://www.aclweb.org/anthology/P15-1168	t	75
2657	Neural machine translation and sequence-to-sequence models: A tutorial	G Neubig	This tutorial introduces a new and powerful set of techniques variously called" neural machine translation" or" neural sequence-to-sequence models". These techniques have been used in a number of tasks regarding the handling of human language, and can be a …	\N	2017	https://arxiv.org/abs/1703.01619	t	75
2658	Nasari: a novel approach to a semantically-aware representation of items	J Camacho-Collados, MT Pilehvar…	The semantic representation of individual word senses and concepts is of fundamental importance to several applications in Natural Language Processing. To date, concept modeling techniques have in the main based their representation either on lexicographic …	\N	2015	https://www.aclweb.org/anthology/N15-1059	t	74
1285	Deep enhanced representation for implicit discourse relation recognition	H Bai, H Zhao	Implicit discourse relation recognition is a challenging task as the relation prediction without explicit connectives in discourse parsing needs understanding of text spans and cannot be easily derived from surface features from the input sentence pairs. Thus, properly …	\N	2018	https://arxiv.org/abs/1807.05154	t	20
2659	Predicting semantically linkable knowledge in developer online forums via convolutional neural network	B Xu, D Ye, Z Xing, X Xia, G Chen, S Li	Consider a question and its answers in Stack Overflow as a knowledge unit. Knowledge units often contain semantically relevant knowledge, and thus linkable for different purposes, such as duplicate questions, directly linkable for problem solving, indirectly linkable for …	\N	2016	https://dl.acm.org/citation.cfm?id=2970357	t	74
2660	Vip-cnn: Visual phrase guided convolutional neural network	Y Li, W Ouyang, X Wang…	As the intermediate level task connecting image captioning and object detection, visual relationship detection started to catch researchers' attention because of its descriptive power and clear structure. It detects the objects and captures their pair-wise interactions with a …	\N	2017	http://openaccess.thecvf.com/content_cvpr_2017/html/Li_ViP-CNN_Visual_Phrase_CVPR_2017_paper.html	t	74
2661	Do characters abuse more than words?	Y Mehdad, J Tetreault	Although word and character n-grams have been used as features in different NLP applications, no systematic comparison or analysis has shown the power of character-based features for detecting abusive language. In this study, we investigate the effectiveness of …	\N	2016	https://www.aclweb.org/anthology/W16-3638	t	74
2662	Audio word2vec: Unsupervised learning of audio segment representations using sequence-to-sequence autoencoder	YA Chung, CC Wu, CH Shen, HY Lee…	The vector representations of fixed dimensionality for words (in text) offered by Word2Vec have been shown to be very useful in many application scenarios, in particular due to the semantic information they carry. This paper proposes a parallel version, the Audio …	\N	2016	https://arxiv.org/abs/1603.00982	t	74
2663	Learning sense-specific word embeddings by exploiting bilingual resources	J Guo, W Che, H Wang, T Liu	Recent work has shown success in learning word embeddings with neural network language models (NNLM). However, the majority of previous NNLMs represent each word with a single embedding, which fails to capture polysemy. In this paper, we address this …	\N	2014	https://www.aclweb.org/anthology/C14-1048	t	73
2664	Multi-cue zero-shot learning with strong supervision	Z Akata, M Malinowski, M Fritz…	Scaling up visual category recognition to large numbers of classes remains challenging. A promising research direction is zero-shot learning, which does not require any training data to recognize new classes, but rather relies on some form of auxiliary information describing …	\N	2016	https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Akata_Multi-Cue_Zero-Shot_Learning_CVPR_2016_paper.html	t	73
2665	Trans-gram, fast cross-lingual word-embeddings	J Coulmance, JM Marty, G Wenzek…	We introduce Trans-gram, a simple and computationally-efficient method to simultaneously learn and align wordembeddings for a variety of languages, using only monolingual data and a smaller set of sentence-aligned data. We use our new method to compute aligned …	\N	2016	https://arxiv.org/abs/1601.02502	t	73
2666	Renoun: Fact extraction for nominal attributes	M Yahya, S Whang, R Gupta, A Halevy	Search engines are increasingly relying on large knowledge bases of facts to provide direct answers to users' queries. However, the construction of these knowledge bases is largely manual and does not scale to the long and heavy tail of facts. Open information extraction …	\N	2014	https://www.aclweb.org/anthology/D14-1038	t	72
2667	Ontologically grounded multi-sense representation learning for semantic vector space models	SK Jauhar, C Dyer, E Hovy	Words are polysemous. However, most approaches to representation learning for lexical semantics assign a single vector to every surface word type. Meanwhile, lexical ontologies such as WordNet provide a source of complementary knowledge to distributional …	\N	2015	https://www.aclweb.org/anthology/N15-1070	t	72
2668	Black holes and white rabbits: Metaphor identification with visual features	E Shutova, D Kiela, J Maillard	Metaphor is pervasive in our communication, which makes it an important problem for natural language processing (NLP). Numerous approaches to metaphor processing have thus been proposed, all of which relied on linguistic features and textual data to construct …	\N	2016	https://www.aclweb.org/anthology/N16-1020	t	72
2669	Adapting word2vec to named entity recognition	SK Sienčnik	In this paper we explore how word vectors built using word2vec can be used to improve the performance of a classifier during Named Entity Recognition. Thereby, we discuss the best integration of word embeddings into the classification problem and consider the effect of the …	\N	2015	http://www.ep.liu.se/ecp/article.asp?issue=109&article=030	t	72
2670	Kelp at semeval-2016 task 3: Learning semantic relations between questions and answers	S Filice, D Croce, A Moschitti, R Basili	This paper describes the KeLP system participating in the SemEval-2016 Community Question Answering (cQA) task. The challenge tasks are modeled as binary classification problems: kernel-based classifiers are trained on the SemEval datasets and their scores are …	\N	2016	https://www.aclweb.org/anthology/S16-1172	t	71
2671	Visual madlibs: Fill in the blank image generation and question answering	L Yu, E Park, AC Berg, TL Berg	In this paper, we introduce a new dataset consisting of 360,001 focused natural language descriptions for 10,738 images. This dataset, the Visual Madlibs dataset, is collected using automatically produced fill-in-the-blank templates designed to gather targeted descriptions …	\N	2015	https://arxiv.org/abs/1506.00278	t	71
2672	Co-learning of word representations and morpheme representations	S Qiu, Q Cui, J Bian, B Gao, TY Liu	The techniques of using neural networks to learn distributed word representations (ie, word embeddings) have been used to solve a variety of natural language processing tasks. The recently proposed methods, such as CBOW and Skip-gram, have demonstrated their …	\N	2014	https://www.aclweb.org/anthology/C14-1015	t	71
2673	Unsupervised word and dependency path embeddings for aspect term extraction	Y Yin, F Wei, L Dong, K Xu, M Zhang…	In this paper, we develop a novel approach to aspect term extraction based on unsupervised learning of distributed representations of words and dependency paths. The basic idea is to connect two words (w1 and w2) with the dependency path (r) between them in the …	\N	2016	https://arxiv.org/abs/1605.07843	t	71
1330	Robust lexical features for improved neural network named-entity recognition	A Ghaddar, P Langlais	Neural network approaches to Named-Entity Recognition reduce the need for carefully hand-crafted features. While some features do remain in state-of-the-art systems, lexical features have been mostly discarded, with the exception of gazetteers. In this work, we show that this …	\N	2018	https://arxiv.org/abs/1806.03489	t	20
748	Cross-lingual dependency parsing based on distributed representations	J Guo, W Che, D Yarowsky, H Wang, T Liu	This paper investigates the problem of **cross-lingual dependency parsing**, aiming at inducing dependency parsers for low-resource languages while using only training data from a resource-rich language (e.g. English).\n\nExisting approaches typically don’t include lexical features, which are not transferable across languages. In this paper, we bridge the lexical feature gap by using distributed feature representations and their composition. We provide two algorithms for **inducing cross-lingual distributed representations of words**, which map vocabularies from two different languages into a common vector space. Consequently, both lexical features and non-lexical features can be used in our model for cross-lingual transfer.\n\nFurthermore, our framework is able to incorporate additional useful features such as cross-lingual word clusters. Our combined contributions achieve an average relative error reduction of 10.9% in labeled attachment score as compared with the delexicalized parser, trained on English universal treebank and transferred to three other languages. It also significantly outperforms McDonald et al. (2013) augmented with projected cluster features on identical data.	\N	2015	https://www.aclweb.org/anthology/P15-1119.pdf	t	98
2674	A roadmap towards machine intelligence	T Mikolov, A Joulin, M Baroni	The development of intelligent machines is one of the biggest unsolved challenges in computer science. In this paper, we propose some fundamental properties these machines should have, focusing in particular on communication and learning. We discuss a simple …	\N	2016	https://link.springer.com/chapter/10.1007/978-3-319-75477-2_2	t	70
2675	Identifying adverse drug event information in clinical notes with distributional semantic representations of context	A Henriksson, M Kvist, H Dalianis, M Duneld	For the purpose of post-marketing drug safety surveillance, which has traditionally relied on the voluntary reporting of individual cases of adverse drug events (ADEs), other sources of information are now being explored, including electronic health records (EHRs), which give …	\N	2015	https://www.sciencedirect.com/science/article/pii/S153204641500180X	t	70
2676	I am robot:(deep) learning to break semantic image captchas	S Sivakorn, I Polakis…	Since their inception, captchas have been widely used for preventing fraudsters from performing illicit actions. Nevertheless, economic incentives have resulted in an arms race, where fraudsters develop automated solvers and, in turn, captcha services tweak their …	\N	2016	https://ieeexplore.ieee.org/abstract/document/7467367/	t	70
2677	Analysis of the effect of sentiment analysis on extracting adverse drug reactions from tweets and forum posts	I Korkontzelos, A Nikfarjam, M Shardlow…	Objective The abundance of text available in social media and health related forums along with the rich expression of public opinion have recently attracted the interest of the public health community to use these sources for pharmacovigilance. Based on the intuition that …	\N	2016	https://www.sciencedirect.com/science/article/pii/S1532046416300508	t	70
2678	A survey on recent advances in named entity recognition from deep learning models	V Yadav, S Bethard	Named Entity Recognition (NER) is a key component in NLP systems for question answering, information retrieval, relation extraction, etc. NER systems have been studied and developed widely for decades, but accurate systems using deep neural networks (NN) …	\N	2019	https://arxiv.org/abs/1910.11470	t	69
2679	How cosmopolitan are emojis?: Exploring emojis usage and meaning over different languages with distributional semantics	F Barbieri, G Kruszewski, F Ronzano…	Choosing the right emoji to visually complement or condense the meaning of a message has become part of our daily life. Emojis are pictures, which are naturally combined with plain text, thus creating a new form of language. These pictures are the same independently …	\N	2016	https://dl.acm.org/citation.cfm?id=2967278	t	69
2680	Issues in evaluating semantic spaces using word analogies	T Linzen	The offset method for solving word analogies has become a standard evaluation tool for vector-space semantic models: it is considered desirable for a space to represent semantic relations as consistent vector offsets. We show that the method's reliance on cosine similarity …	\N	2016	https://arxiv.org/abs/1606.07736	t	69
2681	A model of coherence based on distributed sentence representation	J Li, E Hovy	Coherence is what makes a multi-sentence text meaningful, both logically and syntactically. To solve the challenge of ordering a set of sentences into coherent order, existing approaches focus mostly on defining and using sophisticated features to capture the cross …	\N	2014	https://www.aclweb.org/anthology/D14-1218	t	69
2682	CAPTION-ing the situation: A lexically-derived taxonomy of psychological situation characteristics.	S Parrigon, SE Woo, L Tay, T Wang	In comparison with personality taxonomic research, there has been much less advancement toward establishing an integrative taxonomy of psychological situation characteristics (similar to personality characteristics for persons). One of the main concerns has been the …	\N	2017	https://psycnet.apa.org/record/2016-40096-001	t	69
2683	Morphological word embeddings	R Cotterell, H Schütze	Linguistic similarity is multi-faceted. For instance, two words may be similar with respect to semantics, syntax, or morphology inter alia. Continuous word-embeddings have been shown to capture most of these shades of similarity to some degree. This work considers …	\N	2019	https://arxiv.org/abs/1907.02423	t	68
2492	Groundwater contamination associated with a potential nuclear waste repository at Yucca Mountain, USA	MO Schwartz	The groundwater contamination originating from a potential nuclear waste repository at Yucca Mountain, USA, is evaluated in a three-dimensional flow transport simulation. The model has 833,079 elements and includes both the saturated and unsaturated zone …	\N	2019	https://link.springer.com/article/10.1007/s10064-019-01591-2	t	19
2493	Lxmert: Learning cross-modality encoder representations from transformers	H Tan, M Bansal	Vision-and-language reasoning requires an understanding of visual concepts, language semantics, and, most importantly, the alignment and relationships between these two modalities. We thus propose the LXMERT (Learning Cross-Modality Encoder …	\N	2019	https://arxiv.org/abs/1908.07490	t	19
1298	Seq2seq dependency parsing	Z Li, J Cai, S He, H Zhao	This paper presents a sequence to sequence (seq2seq) dependency parser by directly predicting the relative position of head for each given word, which therefore results in a truly end-to-end seq2seq dependency parser for the first time. Enjoying the advantage of …	\N	2018	https://www.aclweb.org/anthology/C18-1271.pdf	t	19
2684	Integrating distributional lexical contrast into word embeddings for antonym-synonym distinction	KA Nguyen, SS Walde, NT Vu	We propose a novel vector representation that integrates lexical contrast into distributional vectors and strengthens the most salient features for determining degrees of word similarity. The improved vectors significantly outperform standard models and distinguish antonyms …	\N	2016	https://arxiv.org/abs/1605.07766	t	67
2685	Interleaved text/image deep mining on a very large-scale radiology database	HC Shin, L Lu, L Kim, A Seff, J Yao…	Despite tremendous progress in computer vision, effective learning on very large-scale (> 100K patients) medical image databases has been vastly hindered. We present an interleaved text/image deep learning system to extract and mine the semantic interactions of …	\N	2015	http://openaccess.thecvf.com/content_cvpr_2015/html/Shin_Interleaved_TextImage_Deep_2015_CVPR_paper.html	t	66
2686	Automatic detection of cyberbullying on social networks based on bullying features	R Zhao, A Zhou, K Mao	With the increasing use of social media, cyberbullying behaviour has received more and more attention. Cyberbullying may cause many serious and negative impacts on a person's life and even lead to teen suicide. To reduce and stop cyberbullying, one effective solution is …	\N	2016	https://dl.acm.org/citation.cfm?id=2849567	t	66
2687	Evaluating distributed word representations for capturing semantics of biomedical concepts	M TH, S Sahu, A Anand	Recently there is a surge in interest in learning vector representations of words using huge corpus in unsupervised manner. Such word vector representations, also known as word embedding, have been shown to improve the performance of machine learning models in …	\N	2015	https://www.aclweb.org/anthology/W15-3820	t	66
2688	Intrinsic evaluation of word vectors fails to predict extrinsic performance	B Chiu, A Korhonen, S Pyysalo	The quality of word representations is frequently assessed using correlation with human judgements of word similarity. Here, we question whether such intrinsic evaluation can predict the merits of the representations for downstream tasks. We study the correlation …	\N	2016	https://www.aclweb.org/anthology/W16-2501	t	66
2689	A large scale evaluation of distributional semantic models: Parameters, interactions and model selection	G Lapesa, S Evert	This paper presents the results of a large-scale evaluation study of window-based Distributional Semantic Models on a wide variety of tasks. Our study combines a broad coverage of model parameters with a model selection methodology that is robust to …	\N	2014	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00201	t	65
2690	From senses to texts: An all-in-one graph-based approach for measuring semantic similarity	MT Pilehvar, R Navigli	Quantifying semantic similarity between linguistic items lies at the core of many applications in Natural Language Processing and Artificial Intelligence. It has therefore received a considerable amount of research interest, which in its turn has led to a wide range of …	\N	2015	https://www.sciencedirect.com/science/article/pii/S000437021500106X	t	65
2691	Attsum: Joint learning of focusing and summarization with neural attention	Z Cao, W Li, S Li, F Wei, Y Li	Query relevance ranking and sentence saliency ranking are the two main tasks in extractive query-focused summarization. Previous supervised summarization systems often perform the two tasks in isolation. However, since reference summaries are the trade-off between …	\N	2016	https://arxiv.org/abs/1604.00125	t	65
2692	Matrix tri-factorization with manifold regularizations for zero-shot learning	X Xu, F Shen, Y Yang, D Zhang…	Zero-shot learning (ZSL) aims to recognize objects of unseen classes with available training data from another set of seen classes. Existing solutions are focused on exploring knowledge transfer via an intermediate semantic embedding (egs, attributes) shared …	\N	2017	http://openaccess.thecvf.com/content_cvpr_2017/html/Xu_Matrix_Tri-Factorization_With_CVPR_2017_paper.html	t	64
2693	A word embedding approach to predicting the compositionality of multiword expressions	B Salehi, P Cook, T Baldwin	This paper presents the first attempt to use word embeddings to predict the compositionality of multiword expressions. We consider both single-and multi-prototype word embeddings. Experimental results show that, in combination with a back-off method based on string …	\N	2015	https://www.aclweb.org/anthology/N15-1099	t	64
2694	Factoring variations in natural images with deep gaussian mixture models	A Van den Oord, B Schrauwen	Generative models can be seen as the swiss army knives of machine learning, as many problems can be written probabilistically in terms of the distribution of the data, including prediction, reconstruction, imputation and simulation. One of the most promising directions …	\N	2014	http://papers.nips.cc/paper/5227-factoring-variations-in-natural-images-with-deep-gaussian-mixture-models	t	63
2695	Discovering structure in high-dimensional data through correlation explanation	G Ver Steeg, A Galstyan	We introduce a method to learn a hierarchy of successively more abstract representations of complex data based on optimizing an information-theoretic objective. Intuitively, the optimization searches for a set of latent factors that best explain the correlations in the data …	\N	2014	http://papers.nips.cc/paper/5580-discovering-structure-in-high-dimensional-data-through-correlation-explanation	t	61
2696	A simple word embedding model for lexical substitution	O Melamud, O Levy, I Dagan	The lexical substitution task requires identifying meaning-preserving substitutes for a target word instance in a given sentential context. Since its introduction in SemEval-2007, various models addressed this challenge, mostly in an unsupervised setting. In this work we …	\N	2015	https://www.aclweb.org/anthology/W15-1501	t	60
2697	Fisher vectors derived from hybrid gaussian-laplacian mixture models for image annotation	B Klein, G Lev, G Sadeh, L Wolf	In the traditional object recognition pipeline, descriptors are densely sampled over an image, pooled into a high dimensional non-linear representation and then passed to a classifier. In recent years, Fisher Vectors have proven empirically to be the leading …	\N	2014	https://arxiv.org/abs/1411.7399	t	60
2698	Exploring session context using distributed representations of queries and reformulations	B Mitra	Search logs contain examples of frequently occurring patterns of user reformulations of queries. Intuitively, the reformulation" San Francisco"--" San Francisco 49ers" is semantically similar to" Detroit"--" Detroit Lions". Likewise," London"--" things to do in London" and" New …	\N	2015	https://dl.acm.org/citation.cfm?id=2767702	t	60
2699	Developing a successful SemEval task in sentiment analysis of Twitter and other social media texts	P Nakov, S Rosenthal, S Kiritchenko…	We present the development and evaluation of a semantic analysis task that lies at the intersection of two very trendy lines of research in contemporary computational linguistics:(1) sentiment analysis, and (2) natural language processing of social media text. The task was …	\N	2016	https://link.springer.com/article/10.1007/s10579-015-9328-1	t	59
1706	To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks	M Peters, S Ruder, NA Smith	While most previous work has focused on different pretraining objectives and architectures for transfer learning, we ask how to best adapt the pretrained model to a given target task. We focus on the two most common forms of adaptation, feature extraction (where the …	\N	2019	https://arxiv.org/abs/1903.05987	t	31
1299	Interpretation of natural language rules in conversational machine reading	M Saeidi, M Bartolo, P Lewis, S Singh…	Most work in machine reading focuses on question answering problems where the answer is directly expressed in the text to read. However, many real-world question answering problems require the reading of text not because it contains the literal answer, but because it …	\N	2018	https://arxiv.org/abs/1809.01494	t	19
1321	Syntactic scaffolds for semantic structures	S Swayamdipta, S Thomson, K Lee…	We introduce the syntactic scaffold, an approach to incorporating syntactic information into semantic tasks. Syntactic scaffolds avoid expensive syntactic processing at runtime, only making use of a treebank during training, through a multitask objective. We improve over …	\N	2018	https://arxiv.org/abs/1808.10485	t	19
2494	Bert post-training for review reading comprehension and aspect-based sentiment analysis	H Xu, B Liu, L Shu, PS Yu	Question-answering plays an important role in e-commerce as it allows potential customers to actively seek crucial information about products or services to help their purchase decision making. Inspired by the recent success of machine reading comprehension (MRC) …	\N	2019	https://arxiv.org/abs/1904.02232	t	19
2495	Utilizing neural networks and linguistic metadata for early detection of depression indications in text sequences	M Trotzek, S Koitka, CM Friedrich	Depression is ranked as the largest contributor to global disability and is also a major reason for suicide. Still, many individuals suffering from forms of depression are not treated for various reasons. Previous studies have shown that depression also has an effect on …	\N	2018	https://arxiv.org/abs/1804.07000	t	17
2496	Exploring the limits of transfer learning with a unified text-to-text transformer	C Raffel, N Shazeer, A Roberts, K Lee…	Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of …	\N	2019	https://arxiv.org/abs/1910.10683	t	17
1350	Flowqa: Grasping flow in history for conversational machine comprehension	HY Huang, E Choi, W Yih	Conversational machine comprehension requires a deep understanding of the conversation history. To enable traditional, single-turn models to encode the history comprehensively, we introduce Flow, a mechanism that can incorporate intermediate representations generated …	\N	2018	https://arxiv.org/abs/1810.06683	t	17
2497	Convolutional self-attention networks	B Yang, L Wang, D Wong, LS Chao, Z Tu	Self-attention networks (SANs) have drawn increasing interest due to their high parallelization in computation and flexibility in modeling dependencies. SANs can be further enhanced with multi-head attention by allowing the model to attend to information from …	\N	2019	https://arxiv.org/abs/1904.03107	t	17
1318	Language modeling teaches you more syntax than translation does: Lessons learned through auxiliary task analysis	KW Zhang, SR Bowman	Recent work using auxiliary prediction task classifiers to investigate the properties of LSTM representations has begun to shed light on why pretrained representations, like ELMo (Peters et al., 2018) and CoVe (McCann et al., 2017), are so beneficial for neural language …	\N	2018	https://arxiv.org/abs/1809.10040	t	16
1310	Interpreting recurrent and attention-based neural models: a case study on natural language inference	R Ghaeini, XZ Fern, P Tadepalli	Deep learning models have achieved remarkable success in natural language inference (NLI) tasks. While these models are widely explored, they are hard to interpret and it is often unclear how and why they actually work. In this paper, we take a step toward explaining …	\N	2018	https://arxiv.org/abs/1808.03894	t	16
1397	The Fact Extraction and VERification (FEVER) Shared Task	J Thorne, A Vlachos, O Cocarascu…	We present the results of the first Fact Extraction and VERification (FEVER) Shared Task. The task challenged participants to classify whether human-written factoid claims could be Supported or Refuted using evidence retrieved from Wikipedia. We received entries from 23 …	\N	2018	https://arxiv.org/abs/1811.10971	t	16
1707	Modeling Recurrence for Transformer	J Hao, X Wang, B Yang, L Wang, J Zhang…	Recently, the Transformer model that is based solely on attention mechanisms, has advanced the state-of-the-art on various machine translation tasks. However, recent studies reveal that the lack of recurrence hinders its further improvement of translation capacity. In …	\N	2019	https://arxiv.org/abs/1904.03092	t	13
2049	End-to-end sequence labeling via bi-directional lstm-cnns-crf	X Ma, E Hovy	State-of-the-art sequence labeling systems traditionally require large amounts of task-specific knowledge in the form of hand-crafted features and data pre-processing.\n\nIn this paper, we introduce a novel **neural network architecture that benefits from both word- and character-level representations** automatically, by using combination of bidirectional LSTM, CNN and CRF. Our system is truly end-to-end, requiring no feature engineering or data pre-processing, thus making it applicable to a wide range of sequence labeling tasks.\n\nWe evaluate our system on two data sets for two sequence labeling tasks: Penn Treebank WSJ corpus for part-of-speech (POS) tagging and CoNLL 2003 corpus for named entity recognition (NER).\nWe obtain state-of-the-art performance on both the two data: 97.55% accuracy for POS tagging and 91.21% F1 for NER. 	\N	2016	https://arxiv.org/abs/1603.01354	t	981
512	A critical review of recurrent neural networks for sequence learning	ZC Lipton, J Berkowitz, C Elkan	Countless learning tasks require dealing with sequential data. Image captioning, speech synthesis, and music generation all require that a model produce outputs that are sequences. In other domains, such as time series prediction, video analysis, and musical information retrieval, a model must learn from inputs that are sequences. Interactive tasks, such as translating natural language, engaging in dialogue, and controlling a robot, often demand both capabilities.\n\nRecurrent neural networks (RNNs) are connectionist models that capture the dynamics of sequences via cycles in the network of nodes. Unlike standard feedforward neural networks, recurrent networks retain a state that can represent information from an arbitrarily long context window. Although recurrent neural networks have traditionally been difficult to train, and often contain millions of parameters, recent advances in network architectures, optimization techniques, and parallel computation have enabled successful large-scale learning with them.\n\nIn recent years, systems based on long short-term memory (LSTM) and bidirectional (BRNN) architectures have demonstrated ground-breaking performance on tasks as varied as image captioning, language translation, and handwriting recognition.\n\nIn this survey, **we review and synthesize the research that over the past three decades first yielded and then made practical these powerful learning models**. When appropriate, we reconcile conflicting notation and nomenclature. Our goal is to provide a self-contained explication of the state of the art together with a historical perspective and references to primary research.\n\nComment\n----------\n\nNo use cases of embeddings	\N	2015	https://arxiv.org/abs/1506.00019	f	839
2048	SemEval-2016 task 4: Sentiment analysis in Twitter	P Nakov, A Ritter, S Rosenthal, F Sebastiani…	This paper discusses the fourth year of the” Sentiment Analysis in Twitter Task”. SemEval-2016 Task 4 comprises five subtasks, three of which represent a significant departure from previous editions. The first two subtasks are reruns from prior years and ask to predict the …\n\nComment\n----------\n\nNo use cases.	\N	2019	https://arxiv.org/abs/1912.01973	f	758
2059	Bottom-up and top-down attention for image captioning and visual question answering	P Anderson, X He, C Buehler…	Top-down visual attention mechanisms have been used extensively in image captioning and visual question answering (VQA) to enable deeper image understanding through fine-grained analysis and even multiple steps of reasoning. In this work, we propose a combined …	\N	2018	http://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.html	t	720
2060	A decomposable attention model for natural language inference	AP Parikh, O Täckström, D Das, J Uszkoreit	We propose a simple neural architecture for natural language inference. Our approach uses attention to decompose the problem into subproblems that can be solved separately, thus making it trivially parallelizable. On the Stanford Natural Language Inference (SNLI) dataset …	\N	2016	https://arxiv.org/abs/1606.01933	t	501
2498	DREAM: A Challenge Data Set and Models for Dialogue-Based Reading Comprehension	K Sun, D Yu, J Chen, D Yu, Y Choi…	We present DREAM, the first dialogue-based multiple-choice reading comprehension data set. Collected from English as a Foreign Language examinations designed by human experts to evaluate the comprehension level of Chinese learners of English, our data set …	\N	2019	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00264	t	16
2057	Evaluation of output embeddings for fine-grained image classification	Z Akata, S Reed, D Walter, H Lee…	Image classification has advanced significantly in recent years with the availability of large-scale image sets. However, fine-grained classification remains a major challenge due to the annotation cost of large numbers of fine-grained categories.\n\nThis project shows that compelling classification performance can be achieved on such categories even without labeled training data. Given image and class embeddings, we learn a compatibility function such that matching embeddings are assigned a higher score than mismatching ones; zero-shot classification of an image proceeds by finding the label yielding the highest joint compatibility score. We use state-of-the-art image features and focus on different supervised attributes and unsupervised output embeddings either derived from hierarchies or learned from unlabeled text corpora.\n\nWe establish a substantially improved state-of-the-art on the Animals with Attributes and Caltech-UCSD Birds datasets. Most encouragingly, we demonstrate that purely unsupervised output embeddings (learned from Wikipedia and improved with fine-grained text) achieve compelling results, even outperforming the previous supervised state-of-the-art. By combining different output embeddings, we further improve results. 	\N	2015	http://openaccess.thecvf.com/content_cvpr_2015/html/Akata_Evaluation_of_Output_2015_CVPR_paper.html	t	463
2069	Attention-based LSTM for aspect-level sentiment classification	Y Wang, M Huang, L Zhao	Aspect-level sentiment classification is a finegrained task in sentiment analysis. Since it provides more complete and in-depth results, aspect-level sentiment analysis has received much attention these years. In this paper, we reveal that the sentiment polarity of a sentence …	\N	2016	https://www.aclweb.org/anthology/D16-1058	t	458
2063	Revisiting semi-supervised learning with graph embeddings	Z Yang, WW Cohen, R Salakhutdinov	We present a semi-supervised learning framework based on graph embeddings. Given a graph between instances, we train an embedding for each instance to jointly predict the class label and the neighborhood context in the graph. We develop both transductive and …	\N	2016	http://www.jmlr.org/proceedings/papers/v48/yanga16.pdf	t	452
2058	Semantically conditioned lstm-based natural language generation for spoken dialogue systems	TH Wen, M Gasic, N Mrksic, PH Su, D Vandyke…	Natural language generation (NLG) is a critical component of spoken dialogue and it has a significant impact both on usability and perceived quality. Most NLG systems in common use employ rules and heuristics and tend to generate rigid and stylised responses without the …	\N	2015	https://arxiv.org/abs/1508.01745	t	435
2065	Reading wikipedia to answer open-domain questions	D Chen, A Fisch, J Weston, A Bordes	This paper proposes to tackle open-domain question answering using Wikipedia as the unique knowledge source: the answer to any factoid question is a text span in a Wikipedia article. This task of machine reading at scale combines the challenges of document retrieval …	\N	2017	https://arxiv.org/abs/1704.00051	t	434
2068	A deep reinforced model for abstractive summarization	R Paulus, C Xiong, R Socher	Attentional, RNN-based encoder-decoder models for abstractive summarization have achieved good performance on short input and output sequences. For longer documents and summaries however these models often include repetitive and incoherent phrases. We …	\N	2017	https://arxiv.org/abs/1705.04304	t	422
2061	Text understanding from scratch	X Zhang, Y LeCun	This article demontrates that we can apply deep learning to text understanding from character-level inputs all the way up to abstract text concepts, using temporal convolutional networks (ConvNets). We apply ConvNets to various large-scale datasets, including …	\N	2015	https://arxiv.org/abs/1502.01710	t	415
2072	Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering	Y Goyal, T Khot, D Summers-Stay…	Problems at the intersection of vision and language are of significant importance both as challenging research questions and for the rich set of applications they enable. However, inherent structure in our world and bias in our language tend to be a simpler signal for …	\N	2017	http://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html	t	400
2062	Deep unordered composition rivals syntactic methods for text classification	M Iyyer, V Manjunatha, J Boyd-Graber…	Many existing deep learning models for natural language processing tasks focus on learning the compositionality of their inputs, which requires many expensive computations. We present a simple deep neural network that competes with and, in some cases …	\N	2015	https://www.aclweb.org/anthology/P15-1162	t	399
2064	The ubuntu dialogue corpus: A large dataset for research in unstructured multi-turn dialogue systems	R Lowe, N Pow, I Serban, J Pineau	This paper introduces the Ubuntu Dialogue Corpus, a dataset containing almost 1 million multi-turn dialogues, with a total of over 7 million utterances and 100 million words. This provides a unique resource for research into building dialogue managers based on neural …	\N	2015	https://arxiv.org/abs/1506.08909	t	392
545	Distant supervision for relation extraction via piecewise convolutional neural networks	D Zeng, K Liu, Y Chen, J Zhao	Two problems arise when using distant supervision for relation extraction. First, in this method, an already existing knowledge base is heuristically aligned to texts, and the alignment results are treated as labeled data. However, the heuristic alignment can fail …	\N	2015	https://www.aclweb.org/anthology/D15-1203	t	389
2073	Long short-term memory-networks for machine reading	J Cheng, L Dong, M Lapata	In this paper we address the question of how to render sequence-level networks better at handling structured input. We propose a machine reading simulator which processes text incrementally from left to right and performs shallow reasoning with memory and attention …	\N	2016	https://arxiv.org/abs/1601.06733	t	383
2067	A survey on learning to hash	J Wang, T Zhang, N Sebe…	Nearest neighbor search is a problem of finding the data points from the database such that the distances from them to the query point are the smallest. Learning to hash is one of the major solutions to this problem and has been widely studied recently. In this paper, we …	\N	2017	https://ieeexplore.ieee.org/abstract/document/7915742/	t	373
2070	Fast matrix factorization for online recommendation with implicit feedback	X He, H Zhang, MY Kan, TS Chua	This paper contributes improvements on both the effectiveness and efficiency of Matrix Factorization (MF) methods for implicit feedback. We highlight two critical issues of existing works. First, due to the large space of unobserved feedback, most existing works resort to …	\N	2016	https://dl.acm.org/citation.cfm?id=2911489	t	371
2066	Dynamic coattention networks for question answering	C Xiong, V Zhong, R Socher	Several deep learning models have been proposed for question answering. However, due to their single-pass nature, they have no way to recover from local maxima corresponding to incorrect answers. To address this problem, we introduce the Dynamic Coattention Network …	\N	2016	https://arxiv.org/abs/1611.01604	t	353
2085	A broad-coverage challenge corpus for sentence understanding through inference	A Williams, N Nangia, SR Bowman	This paper introduces the Multi-Genre Natural Language Inference (MultiNLI) corpus, a dataset designed for use in the development and evaluation of machine learning models for sentence understanding. In addition to being one of the largest corpora available for the task …	\N	2017	https://arxiv.org/abs/1704.05426	t	349
2071	On deep multi-view representation learning	W Wang, R Arora, K Livescu, J Bilmes	We consider learning representations (features) in the setting in which we have access to multiple unlabeled views of the data for representation learning while only one view is available at test time. Previous work on this problem has proposed several techniques …	\N	2015	http://www.jmlr.org/proceedings/papers/v37/wangb15.pdf	t	342
541	Heterogeneous network embedding via deep architectures	S Chang, W Han, J Tang, GJ Qi, CC Aggarwal…	Data embedding is used in many machine learning applications to create low-dimensional feature representations, which preserves the structure of data points in their original space. In this paper, we examine the scenario of a heterogeneous network with nodes and content …	\N	2015	https://dl.acm.org/citation.cfm?id=2783296	t	326
2076	Learning deep representations of fine-grained visual descriptions	S Reed, Z Akata, H Lee…	State-of-the-art methods for zero-shot visual recognition formulate learning as a joint embedding problem of images and side information. In these formulations the current best complement to visual features are attributes: manually-encoded vectors describing shared …	\N	2016	http://openaccess.thecvf.com/content_cvpr_2016/html/Reed_Learning_Deep_Representations_CVPR_2016_paper.html	t	321
577	Neural network methods for natural language processing	Y Goldberg	Neural networks are a family of powerful machine learning models. This book focuses on the application of neural network models to natural language data. The first half of the book (Parts I and II) covers the basics of supervised machine learning and feed-forward neural …	\N	2017	https://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037	t	321
614	Learned in translation: Contextualized word vectors	B McCann, J Bradbury, C Xiong…	Computer vision has benefited from initializing multiple deep layers with weights pretrained on large supervised training sets like ImageNet. Natural language processing (NLP) typically sees initialization of only the lowest layer of deep models with pretrained word …	\N	2017	http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors	t	316
2075	Evaluation methods for unsupervised word embeddings	T Schnabel, I Labutov, D Mimno…	We present a comprehensive study of evaluation methods for unsupervised embedding techniques that obtain meaningful representations of words from text. Different evaluations result in different orderings of embedding methods, calling into question the common …	\N	2015	https://www.aclweb.org/anthology/D15-1036	t	314
2077	Machine comprehension using match-lstm and answer pointer	S Wang, J Jiang	Machine comprehension of text is an important problem in natural language processing. A recently released dataset, the Stanford Question Answering Dataset (SQuAD), offers a large number of real questions and their answers created by humans through crowdsourcing …	\N	2016	https://arxiv.org/abs/1608.07905	t	298
2078	Towards universal paraphrastic sentence embeddings	J Wieting, M Bansal, K Gimpel, K Livescu	We consider the problem of learning general-purpose, paraphrastic sentence embeddings based on supervision from the Paraphrase Database (Ganitkevitch et al., 2013). We compare six compositional architectures, evaluating them on annotated textual similarity …	\N	2015	https://arxiv.org/abs/1511.08198	t	296
2084	Deep neural networks for learning graph representations	S Cao, W Lu, Q Xu	In this paper, we propose a novel model for learning graph representations, which generates a low-dimensional vector representation for each vertex by capturing the graph structural information. Different from other previous research efforts, we adopt a random …	\N	2016	https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12423	t	293
2080	A deep relevance matching model for ad-hoc retrieval	J Guo, Y Fan, Q Ai, WB Croft	In recent years, deep neural networks have led to exciting breakthroughs in speech recognition, computer vision, and natural language processing (NLP) tasks. However, there have been few positive results of deep models on ad-hoc retrieval tasks. This is partially due …	\N	2016	https://dl.acm.org/citation.cfm?id=2983769	t	287
2083	Latent embeddings for zero-shot classification	Y Xian, Z Akata, G Sharma, Q Nguyen…	We present a novel latent embedding model for learning a compatibility function between image and class embeddings, in the context of zero-shot classification. The proposed method augments the state-of-the-art bilinear compatibility model by incorporating latent …	\N	2016	https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Xian_Latent_Embeddings_for_CVPR_2016_paper.html	t	282
624	Conceptnet 5.5: An open multilingual graph of general knowledge	R Speer, J Chin, C Havasi	Machine learning about language can be improved by supplying it with specific knowledge and sources of external information. We present here a new version of the linked open data resource ConceptNet that is particularly well suited to be used with modern NLP …	\N	2017	https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14972	t	280
2089	Aspect level sentiment classification with deep memory network	D Tang, B Qin, T Liu	We introduce a deep memory network for aspect level sentiment classification. Unlike feature-based SVM and sequential neural models such as LSTM, this approach explicitly captures the importance of each context word when inferring the sentiment polarity of an …	\N	2016	https://arxiv.org/abs/1605.08900	t	271
2081	Achieving open vocabulary neural machine translation with hybrid word-character models	MT Luong, CD Manning	Nearly all previous work on neural machine translation (NMT) has used quite restricted vocabularies, perhaps with a subsequent method to patch in unknown words. This paper presents a novel word-character solution to achieving open vocabulary NMT. We build …	\N	2016	https://arxiv.org/abs/1604.00788	t	264
2088	Convolutional matrix factorization for document context-aware recommendation	D Kim, C Park, J Oh, S Lee, H Yu	Sparseness of user-to-item rating data is one of the major factors that deteriorate the quality of recommender system. To handle the sparsity problem, several recommendation techniques have been proposed that additionally consider auxiliary information to improve …	\N	2016	https://dl.acm.org/citation.cfm?id=2959165	t	263
2110	Qanet: Combining local convolution with global self-attention for reading comprehension	AW Yu, D Dohan, MT Luong, R Zhao, K Chen…	Current end-to-end machine reading and question answering (Q\\&A) models are primarily based on recurrent neural networks (RNNs) with attention. Despite their success, these models are often slow for both training and inference due to the sequential nature of RNNs …	\N	2018	https://arxiv.org/abs/1804.09541	t	251
2082	Bilbowa: Fast bilingual distributed representations without word alignments	S Gouws, Y Bengio, G Corrado	We introduce BilBOWA (Bilingual Bag-of-Words without Alignments), a simple and computationally-efficient model for learning bilingual distributed representations of words which can scale to large monolingual datasets and does not require word-aligned parallel …	\N	2015	http://www.jmlr.org/proceedings/papers/v37/gouws15.pdf	t	249
2087	Bilateral multi-perspective matching for natural language sentences	Z Wang, W Hamza, R Florian	Natural language sentence matching is a fundamental technology for a variety of tasks. Previous approaches either match sentences from a single direction or only apply single granular (word-by-word or sentence-by-sentence) matching. In this work, we propose a …	\N	2017	https://arxiv.org/abs/1702.03814	t	242
2099	Effective LSTMs for target-dependent sentiment classification	D Tang, B Qin, X Feng, T Liu	Target-dependent sentiment classification remains a challenge: modeling the semantic relatedness of a target with its context words in a sentence. Different context words have different influences on determining the sentiment polarity of a sentence towards the target …	\N	2015	https://arxiv.org/abs/1512.01100	t	238
571	Multi-perspective sentence similarity modeling with convolutional neural networks	H He, K Gimpel, J Lin	Modeling sentence similarity is complicated by the ambiguity and variability of linguistic expression. To cope with these challenges, we propose a model for comparing sentences that uses a multiplicity of perspectives. We first model each sentence using a convolutional …	\N	2015	https://www.aclweb.org/anthology/D15-1181	t	233
2103	Deep learning for hate speech detection in tweets	P Badjatiya, S Gupta, M Gupta, V Varma	Hate speech detection on Twitter is critical for applications like controversial event extraction, building AI chatterbots, content recommendation, and sentiment analysis. We define this task as being able to classify a tweet as racist, sexist or neither. The complexity of …	\N	2017	https://dl.acm.org/citation.cfm?id=3054223	t	233
2106	Zero-shot learning-a comprehensive evaluation of the good, the bad and the ugly	Y Xian, CH Lampert, B Schiele…	Due to the importance of zero-shot learning, ie classifying images where there is a lack of labeled training data, the number of proposed approaches has recently increased steadily. We argue that it is time to take a step back and to analyze the status quo of the area. The …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8413121/	t	230
2095	Harnessing deep neural networks with logic rules	Z Hu, X Ma, Z Liu, E Hovy, E Xing	Combining deep neural networks with structured logic rules is desirable to harness flexibility and reduce uninterpretability of the neural models. We propose a general framework capable of enhancing various types of neural networks (eg, CNNs and RNNs) with …	\N	2016	https://arxiv.org/abs/1603.06318	t	227
2104	Poincaré embeddings for learning hierarchical representations	M Nickel, D Kiela	Representation learning has become an invaluable approach for learning from symbolic data such as text and graphs. However, state-of-the-art embedding methods typically do not account for latent hierarchical structures which are characteristic for many …	\N	2017	http://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hie	t	226
2086	Long short-term memory over recursive structures	X Zhu, P Sobihani, H Guo	The chain-structured long short-term memory (LSTM) has showed to be effective in a wide range of problems such as speech recognition and machine translation. In this paper, we propose to extend it to tree structures, in which a memory cell can reflect the history …	\N	2015	http://www.jmlr.org/proceedings/papers/v37/zhub15.pdf	t	221
2098	A joint many-task model: Growing a neural network for multiple nlp tasks	K Hashimoto, C Xiong, Y Tsuruoka…	Transfer and multi-task learning have traditionally focused on either a single source-target pair or very few, similar tasks. Ideally, the linguistic levels of morphology, syntax and semantics would benefit each other by being trained in a single model. We introduce a joint …	\N	2016	https://arxiv.org/abs/1611.01587	t	216
2112	Semi-supervised sequence tagging with bidirectional language models	ME Peters, W Ammar, C Bhagavatula…	Pre-trained word embeddings learned from unlabeled text have become a standard component of neural network architectures for NLP tasks. However, in most cases, the recurrent network that operates on word-level representations to produce context sensitive …	\N	2017	https://arxiv.org/abs/1705.00108	t	216
2091	Learning natural language inference with LSTM	S Wang, J Jiang	Natural language inference (NLI) is a fundamentally important task in natural language processing that has many applications. The recently released Stanford Natural Language Inference (SNLI) corpus has made it possible to develop and evaluate learning-centered …	\N	2015	https://arxiv.org/abs/1512.08849	t	215
2094	Learning to generate reviews and discovering sentiment	A Radford, R Jozefowicz, I Sutskever	We explore the properties of byte-level recurrent language models. When given sufficient amounts of capacity, training data, and compute time, the representations learned by these models include disentangled features corresponding to high-level concepts. Specifically, we …	\N	2017	https://arxiv.org/abs/1704.01444	t	214
2090	Representation learning using multi-task deep neural networks for semantic classification and information retrieval	X Liu, J Gao, X He, L Deng, K Duh, YY Wang	Methods of deep neural networks (DNNs) have recently demonstrated superior performance on a number of natural language processing tasks. However, in most previous work, the models are learned based on either unsupervised objectives, which does not directly …	\N	2015	https://www.microsoft.com/en-us/research/publication/representation-learning-using-multi-task-deep-neural-networks-for-semantic-classification-and-information-retrieval/	t	213
600	Sequential short-text classification with recurrent and convolutional neural networks	JY Lee, F Dernoncourt	Recent approaches based on artificial neural networks (ANNs) have shown promising results for short-text classification. However, many short texts occur in sequences (eg, sentences in a document or utterances in a dialog), and most existing ANN-based systems …	\N	2016	https://arxiv.org/abs/1603.03827	t	213
650	Unsupervised learning of sentence embeddings using compositional n-gram features	M Pagliardini, P Gupta, M Jaggi	The recent tremendous success of unsupervised word embeddings in a multitude of applications raises the obvious question if similar methods could be derived to improve embeddings (ie semantic representations) of word sequences as well. We present a simple …	\N	2017	https://arxiv.org/abs/1703.02507	t	211
2093	A fast unified model for parsing and sentence understanding	SR Bowman, J Gauthier, A Rastogi, R Gupta…	Tree-structured neural networks exploit valuable syntactic parse information as they interpret the meanings of sentences. However, they suffer from two key technical problems that make them slow and unwieldy for large-scale NLP tasks: they usually operate on parsed …	\N	2016	https://arxiv.org/abs/1603.06021	t	208
2116	Anchors: High-precision model-agnostic explanations	MT Ribeiro, S Singh, C Guestrin	We introduce a novel model-agnostic system that explains the behavior of complex models with high-precision rules called anchors, representing local," sufficient" conditions for predictions. We propose an algorithm to efficiently compute these explanations for any black …	\N	2018	https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16982	t	207
629	Multi-layer representation learning for medical concepts	E Choi, MT Bahadori, E Searles, C Coffey…	Proper representations of medical concepts such as diagnosis, medication, procedure codes and visits from Electronic Health Records (EHR) has broad applications in healthcare analytics. Patient EHR data consists of a sequence of visits over time, where each visit …	\N	2016	https://dl.acm.org/citation.cfm?id=2939823	t	205
2092	Learning semantic representations of users and products for document level sentiment classification	D Tang, B Qin, T Liu	Neural network methods have achieved promising results for sentiment classification of text. However, these models only use semantics of texts, while ignoring users who express the sentiment and products which are evaluated, both of which have great influences on …	\N	2015	https://www.aclweb.org/anthology/P15-1098	t	201
572	Language understanding for text-based games using deep reinforcement learning	K Narasimhan, T Kulkarni, R Barzilay	In this paper, we consider the task of learning control policies for text-based games. In these games, all interactions in the virtual world are through text and the underlying state is not observed. The resulting language barrier makes such environments challenging for …	\N	2015	https://arxiv.org/abs/1506.08941	t	201
2096	Gated-attention readers for text comprehension	B Dhingra, H Liu, Z Yang, WW Cohen…	In this paper we study the problem of answering cloze-style questions over documents. Our model, the Gated-Attention (GA) Reader, integrates a multi-hop architecture with a novel attention mechanism, which is based on multiplicative interactions between the query …	\N	2016	https://arxiv.org/abs/1606.01549	t	200
680	Deep biaffine attention for neural dependency parsing	T Dozat, CD Manning	This paper builds off recent work from Kiperwasser & Goldberg (2016) using neural attention in a simple graph-based dependency parser. We use a larger but more thoroughly regularized parser than other recent BiLSTM-based approaches, with biaffine classifiers to …	\N	2016	https://arxiv.org/abs/1611.01734	t	200
588	Generalized low rank models	M Udell, C Horn, R Zadeh, S Boyd	Principal components analysis (PCA) is a well-known technique for approximating a tabular data set by a low rank matrix. Here, we extend the idea of PCA to handle arbitrary data sets consisting of numerical, Boolean, categorical, ordinal, and other data types. This framework …	\N	2016	http://www.nowpublishers.com/article/Details/MAL-055	t	198
580	Autoextend: Extending word embeddings to embeddings for synsets and lexemes	S Rothe, H Schütze	We present\\textit {AutoExtend}, a system to learn embeddings for synsets and lexemes. It is flexible in that it can take any word embeddings as input and does not need an additional training corpus. The synset/lexeme embeddings obtained live in the same vector space as …	\N	2015	https://arxiv.org/abs/1507.01127	t	197
593	Bilingual word representations with monolingual quality in mind	T Luong, H Pham, CD Manning	Recent work in learning bilingual representations tend to tailor towards achieving good performance on bilingual tasks, most often the crosslingual document classification (CLDC) evaluation, but to the detriment of preserving clustering structures of word representations …	\N	2015	https://www.aclweb.org/anthology/W15-1521	t	193
490	Line: Large-scale information network embedding	J Tang, M Qu, M Wang, M Zhang, J Yan…	This paper studies the problem of embedding very large information networks into low-dimensional vector spaces, which is useful in many tasks such as visualization, node classification, and link prediction. Most existing graph embedding methods do not scale for …	\N	2015	https://dl.acm.org/citation.cfm?id=2741093	f	1805
513	On using very large target vocabulary for neural machine translation	S Jean, K Cho, R Memisevic, Y Bengio	Neural machine translation, a recently proposed approach to machine translation based purely on neural networks, has shown promising results compared to the existing approaches such as phrase-based statistical machine translation. Despite its recent …\n\nComment\n----------\n\nNot about WEMs.	\N	2014	https://arxiv.org/abs/1412.2007	f	650
2101	Quasi-recurrent neural networks	J Bradbury, S Merity, C Xiong, R Socher	Recurrent neural networks are a powerful tool for modeling sequential data, but the dependence of each timestep's computation on the previous timestep's output limits parallelism and makes RNNs unwieldy for very long sequences. We introduce quasi …	\N	2016	https://arxiv.org/abs/1611.01576	t	190
604	Semeval-2016 task 1: Semantic textual similarity, monolingual and cross-lingual evaluation	E Agirre, C Banea, D Cer, M Diab…	Semantic Textual Similarity (STS) seeks to measure the degree of semantic equivalence between two snippets of text. Similarity is expressed on an ordinal scale that spans from semantic equivalence to complete unrelatedness. Intermediate values capture …	\N	2016	https://www.aclweb.org/anthology/S16-1081	t	187
687	Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation	D Cer, M Diab, E Agirre, I Lopez-Gazpio…	Semantic Textual Similarity (STS) measures the meaning similarity of sentences. Applications include machine translation (MT), summarization, generation, question answering (QA), short answer grading, semantic search, dialog and conversational systems …	\N	2017	https://arxiv.org/abs/1708.00055	t	184
2124	End-to-end neural coreference resolution	K Lee, L He, M Lewis, L Zettlemoyer	We introduce the first end-to-end coreference resolution model and show that it significantly outperforms all previous work without using a syntactic parser or hand-engineered mention detector. The key idea is to directly consider all spans in a document as potential mentions …	\N	2017	https://arxiv.org/abs/1707.07045	t	182
2100	Reasonet: Learning to stop reading in machine comprehension	Y Shen, PS Huang, J Gao, W Chen	Teaching a computer to read and answer general questions pertaining to a document is a challenging yet unsolved problem. In this paper, we describe a novel neural network architecture called the Reasoning Network (ReasoNet) for machine comprehension tasks …	\N	2017	https://dl.acm.org/citation.cfm?id=3098177	t	180
606	Suggesting accurate method and class names	M Allamanis, ET Barr, C Bird, C Sutton	Descriptive names are a vital part of readable, and hence maintainable, code. Recent progress on automatically suggesting names for local variables tantalizes with the prospect of replicating that success with method and class names. However, suggesting names for …	\N	2015	https://dl.acm.org/citation.cfm?id=2786849	t	180
603	Monolingual and cross-lingual information retrieval models based on (bilingual) word embeddings	I Vulić, MF Moens	We propose a new unified framework for monolingual (MoIR) and cross-lingual information retrieval (CLIR) which relies on the induction of dense real-valued word vectors known as word embeddings (WE) from comparable data. To this end, we make several important …	\N	2015	https://dl.acm.org/citation.cfm?id=2767752	t	179
2102	Tying word vectors and word classifiers: A loss framework for language modeling	H Inan, K Khosravi, R Socher	Recurrent neural networks have been very successful at predicting sequences of words in tasks such as language modeling. However, all such models are based on the conventional classification framework, where the model is trained against one-hot targets, and each word …	\N	2016	https://arxiv.org/abs/1611.01462	t	179
2113	Adversarial multi-task learning for text classification	P Liu, X Qiu, X Huang	Neural network models have shown their promising opportunities for multi-task learning, which focus on learning the shared layers to extract the common and task-invariant features. However, in most existing approaches, the extracted shared features are prone to be …	\N	2017	https://arxiv.org/abs/1704.05742	t	179
2105	Joint learning of character and word embeddings	X Chen, L Xu, Z Liu, M Sun, H Luan	Most word embedding methods take a word as a basic unit and learn embeddings according to words' external contexts, ignoring the internal structures of words. However, in some languages such as Chinese, a word is usually composed of several characters and …	\N	2015	https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/viewPaper/11000	t	178
2097	Traversing knowledge graphs in vector space	K Guu, J Miller, P Liang	Path queries on a knowledge graph can be used to answer compositional questions such as" What languages are spoken by people living in Lisbon?". However, knowledge graphs often have missing facts (edges) which disrupts path queries. Recent models for knowledge …	\N	2015	https://arxiv.org/abs/1506.01094	t	177
2123	Learning discourse-level diversity for neural dialog models using conditional variational autoencoders	T Zhao, R Zhao, M Eskenazi	While recent neural encoder-decoder models have shown great promise in modeling open-domain conversations, they often generate dull and generic responses. Unlike past work that has focused on diversifying the output of the decoder at word-level to alleviate this …	\N	2017	https://arxiv.org/abs/1703.10960	t	177
655	context2vec: Learning generic context embedding with bidirectional lstm	O Melamud, J Goldberger, I Dagan	Context representations are central to various NLP tasks, such as word sense disambiguation, named entity recognition, coreference resolution, and many more. In this work we present a neural model for efficiently learning a generic context embedding function …	\N	2016	https://www.aclweb.org/anthology/K16-1006	t	175
2120	Newsqa: A machine comprehension dataset	A Trischler, T Wang, X Yuan, J Harris, A Sordoni…	We present NewsQA, a challenging machine comprehension dataset of over 100,000 human-generated question-answer pairs. Crowdworkers supply questions and answers based on a set of over 10,000 news articles from CNN, with answers consisting of spans of …	\N	2016	https://arxiv.org/abs/1611.09830	t	173
2131	Survey of the state of the art in natural language generation: Core tasks, applications and evaluation	A Gatt, E Krahmer	This paper surveys the current state of the art in Natural Language Generation (NLG), defined as the task of generating text or speech from non-linguistic input. A survey of NLG is timely in view of the changes that the field has undergone over the past two decades …	\N	2018	http://www.jair.org/papers/paper5477.html	t	173
481	Sequence to sequence learning with neural networks	I Sutskever, O Vinyals, QV Le	Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this …\n\nComment\n----------\n\nNot about embeddings, no use cases.	\N	2014	https://www.arxiv-vanity.com/papers/1409.3215/	f	9015
503	Learning entity and relation embeddings for knowledge graph completion	Y Lin, Z Liu, M Sun, Y Liu, X Zhu	Knowledge graph completion aims to perform link prediction between entities. In this paper, we consider the approach of knowledge graph embeddings. Recently, models such as TransE and TransH build entity and relation embeddings by regarding a relation as …\n\nComment\n----------\n\nNot about word embeddings.	\N	2015	https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9571	f	941
504	word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method	Y Goldberg, O Levy	The word2vec software of Tomas Mikolov and colleagues (this https URL) has gained a lot of traction lately, and provides state-of-the-art word embeddings. The learning models behind the software are described in two research papers. We found the description of the …\n\nComment\n----------\n\nNo use cases, only theoretical background.	\N	2014	https://arxiv.org/abs/1402.3722	f	870
524	Learning character-level representations for part-of-speech tagging	CD Santos, B Zadrozny	Distributed word representations have recently been proven to be an invaluable resource for NLP. These representations are normally learned using neural networks and capture syntactic and semantic information about words. Information about word morphology and …	\N	2014	http://www.jmlr.org/proceedings/papers/v32/santos14.pdf	t	423
525	Learning word embeddings efficiently with noise-contrastive estimation	A Mnih, K Kavukcuoglu	Continuous-valued word embeddings learned by neural language models have recently been shown to capture semantic and syntactic information about words very well, setting performance records on several word similarity tasks. The best results are obtained by …	\N	2013	http://papers.nips.cc/paper/5165-learning-word-embeddings	t	401
2117	Deep semantic role labeling: What works and what's next	L He, K Lee, M Lewis, L Zettlemoyer	We introduce a new deep learning model for semantic role labeling (SRL) that significantly improves the state of the art, along with detailed analyses to reveal its strengths and limitations. We use a deep highway BiLSTM architecture with constrained decoding, while …	\N	2017	https://www.aclweb.org/anthology/papers/P/P17/P17-1044/	t	172
2109	Text as data	M Gentzkow, B Kelly, M Taddy	An ever increasing share of human interaction, communication, and culture is recorded as digital text. We provide an introduction to the use of text as an input to economic research. We discuss the features that make text different from other forms of data, offer a practical …	\N	2019	https://www.aeaweb.org/doi/10.1257/jel.20181020	t	170
631	Query expansion with locally-trained word embeddings	F Diaz, B Mitra, N Craswell	Continuous space word embeddings have received a great deal of attention in the natural language processing and machine learning communities for their ability to model term similarity and other relationships. We study the use of term relatedness in the context of …	\N	2016	https://arxiv.org/abs/1605.07891	t	168
2111	Reporting score distributions makes a difference: Performance study of lstm-networks for sequence tagging	N Reimers, I Gurevych	In this paper we show that reporting a single performance score is insufficient to compare non-deterministic approaches. We demonstrate for common sequence tagging tasks that the seed value for the random number generator can result in statistically significant (p< 10^-4) …	\N	2017	https://arxiv.org/abs/1707.09861	t	166
659	A review of natural language processing techniques for opinion mining systems	S Sun, C Luo, J Chen	As the prevalence of social media on the Internet, opinion mining has become an essential approach to analyzing so many data. Various applications appear in a wide range of industrial domains. Meanwhile, opinions have diverse expressions which bring along …	\N	2017	https://www.sciencedirect.com/science/article/pii/S1566253516301117	t	163
2130	Seq2sql: Generating structured queries from natural language using reinforcement learning	V Zhong, C Xiong, R Socher	A significant amount of the world's knowledge is stored in relational databases. However, the ability for users to retrieve facts from a database is limited due to a lack of understanding of query languages such as SQL. We propose Seq2SQL, a deep neural network for …	\N	2017	https://arxiv.org/abs/1709.00103	t	162
633	From group to individual labels using deep features	D Kotzias, M Denil, N De Freitas, P Smyth	In many classification problems labels are relatively scarce. One context in which this occurs is where we have labels for groups of instances but not for the instances themselves, as in multi-instance learning. Past work on this problem has typically focused on learning …	\N	2015	https://dl.acm.org/citation.cfm?id=2783380	t	161
628	From paraphrase database to compositional paraphrase model and back	J Wieting, M Bansal, K Gimpel, K Livescu	The Paraphrase Database (PPDB; Ganitkevitch et al., 2013) is an extensive semantic resource, consisting of a list of phrase pairs with (heuristic) confidence estimates. However, it is still unclear how it can best be used, due to the heuristic nature of the confidences and …	\N	2015	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00143	t	161
656	Problems with evaluation of word embeddings using word similarity tasks	M Faruqui, Y Tsvetkov, P Rastogi, C Dyer	Lacking standardized extrinsic evaluation methods for vector representations of words, the NLP community has relied heavily on word similarity tasks as a proxy for intrinsic evaluation of word vectors. Word similarity evaluation, which correlates the distance between vectors …	\N	2016	https://arxiv.org/abs/1605.02276	t	158
647	Semantic expansion using word embedding clustering and convolutional neural network for improving short text classification	P Wang, B Xu, J Xu, G Tian, CL Liu, H Hao	Text classification can help users to effectively handle and exploit useful information hidden in large-scale documents. However, the sparsity of data and the semantic sensitivity to context often hinder the classification performance of short texts. In order to overcome the …	\N	2016	https://www.sciencedirect.com/science/article/pii/S0925231215014502	t	157
2133	Text classification improved by integrating bidirectional LSTM with two-dimensional max pooling	P Zhou, Z Qi, S Zheng, J Xu, H Bao, B Xu	Recurrent Neural Network (RNN) is one of the most popular architectures used in Natural Language Processsing (NLP) tasks because its recurrent structure is very suitable to process variable-length text. RNN can utilize distributed representations of words by first …	\N	2016	https://arxiv.org/abs/1611.06639	t	156
2499	“Bilingual Expert” Can Find Translation Errors	K Fan, J Wang, B Li, F Zhou, B Chen, L Si	The performances of machine translation (MT) systems are usually evaluated by the metric BLEU when the golden references are provided. However, in the case of model inference or production deployment, golden references are usually expensively available, such as …	\N	2019	https://www.aaai.org/ojs/index.php/AAAI/article/view/4599	t	15
1301	Glomo: Unsupervisedly learned relational graphs as transferable representations	Z Yang, J Zhao, B Dhingra, K He, WW Cohen…	Modern deep transfer learning approaches have mainly focused on learning generic feature vectors from one task that are transferable to other tasks, such as word embeddings in language and pretrained convolutional features in vision. However, these approaches …	\N	2018	https://arxiv.org/abs/1806.05662	t	15
531	A structured self-attentive sentence embedding	Z Lin, M Feng, CN Santos, M Yu, B Xiang…	This paper proposes a new model for extracting an interpretable sentence embedding by introducing self-attention. Instead of using a vector, we use a 2-D matrix to represent the embedding, with each row of the matrix attending on a different part of the sentence. We also …	\N	2017	https://arxiv.org/abs/1703.03130	t	632
2115	Structured attention networks	Y Kim, C Denton, L Hoang, AM Rush	Attention networks have proven to be an effective approach for embedding categorical inference within a deep neural network. However, for many tasks we may want to model richer structural dependencies without abandoning end-to-end training. In this work, we …	\N	2017	https://arxiv.org/abs/1702.00887	t	153
2107	Deep visual analogy-making	SE Reed, Y Zhang, Y Zhang, H Lee	In addition to identifying the content within a single image, relating images and generating related images are critical tasks for image understanding. Recently, deep convolutional networks have yielded breakthroughs in producing image labels, annotations and captions …	\N	2015	http://papers.nips.cc/paper/5845-deep-visual-analogy-making	t	151
640	Do multi-sense embeddings improve natural language understanding?	J Li, D Jurafsky	Learning a distinct representation for each sense of an ambiguous word could lead to more powerful and fine-grained models of vector-space representations. Yet whilemulti-sense'methods have been proposed and tested on artificial word-similarity tasks, we don't …	\N	2015	https://arxiv.org/abs/1506.01070	t	151
2164	Race: Large-scale reading comprehension dataset from examinations	G Lai, Q Xie, H Liu, Y Yang, E Hovy	We present RACE, a new dataset for benchmark evaluation of methods in the reading comprehension task. Collected from the English exams for middle and high school Chinese students in the age range between 12 to 18, RACE consists of near 28,000 passages and …	\N	2017	https://arxiv.org/abs/1704.04683	t	151
636	PPDB 2.0: Better paraphrase ranking, fine-grained entailment relations, word embeddings, and style classification	E Pavlick, P Rastogi, J Ganitkevitch…	We present a new release of the Paraphrase Database. PPDB 2.0 includes a discriminatively re-ranked set of paraphrases that achieve a higher correlation with human judgments than PPDB 1.0's heuristic rankings. Each paraphrase pair in the database now …	\N	2015	https://www.aclweb.org/anthology/P15-2070	t	147
2118	Neural symbolic machines: Learning semantic parsers on freebase with weak supervision	C Liang, J Berant, Q Le, KD Forbus, N Lao	Harnessing the statistical power of neural networks to perform language understanding and symbolic reasoning is difficult, when it requires executing efficient discrete operations against a large knowledge-base. In this work, we introduce a Neural Symbolic Machine …	\N	2016	https://arxiv.org/abs/1611.00020	t	146
2121	Stance and sentiment in tweets	SM Mohammad, P Sobhani, S Kiritchenko	We can often detect from a person's utterances whether he or she is in favor of or against a given target entity—one's stance toward the target. However, a person may express the same stance toward a target by using negative or positive language. Here for the first time …	\N	2017	https://dl.acm.org/citation.cfm?id=3003433	t	145
2127	Learning natural language inference using bidirectional LSTM model and inner-attention	Y Liu, C Sun, L Lin, X Wang	In this paper, we proposed a sentence encoding-based model for recognizing text entailment. In our approach, the encoding of sentence is a two-stage process. Firstly, average pooling was used over word-level bidirectional LSTM (biLSTM) to generate a first …	\N	2016	https://arxiv.org/abs/1605.09090	t	145
2146	Multi-modal factorized bilinear pooling with co-attention learning for visual question answering	Z Yu, J Yu, J Fan, D Tao	Visual question answering (VQA) is challenging because it requires a simultaneous understanding of both the visual content of images and the textual content of questions. The approaches used to represent the images and questions in a fine-grained manner and …	\N	2017	http://openaccess.thecvf.com/content_iccv_2017/html/Yu_Multi-Modal_Factorized_Bilinear_ICCV_2017_paper.html	t	144
2108	Multi-task deep visual-semantic embedding for video thumbnail selection	W Liu, T Mei, Y Zhang, C Che, J Luo	Given the tremendous growth of online videos, video thumbnail, as the common visualization form of video content, is becoming increasingly important to influence user's browsing and searching experience. However, conventional methods for video thumbnail …	\N	2015	https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Liu_Multi-Task_Deep_Visual-Semantic_2015_CVPR_paper.html	t	142
2143	Datastories at semeval-2017 task 4: Deep lstm with attention for message-level and topic-based sentiment analysis	C Baziotis, N Pelekis, C Doulkeridis	In this paper we present two deep-learning systems that competed at SemEval-2017 Task 4 “Sentiment Analysis in Twitter”. We participated in all subtasks for English tweets, involving message-level and topic-based sentiment polarity classification and quantification. We use …	\N	2017	https://www.aclweb.org/anthology/papers/S/S17/S17-2126/	t	142
2136	Neural belief tracker: Data-driven dialogue state tracking	N Mrkšić, DO Séaghdha, TH Wen, B Thomson…	One of the core components of modern spoken dialogue systems is the belief tracker, which estimates the user's goal at every step of the dialogue. However, most current approaches have difficulty scaling to larger, more complex dialogue domains. This is due to their …	\N	2016	https://arxiv.org/abs/1606.03777	t	141
1413	Context-aware self-attention networks	B Yang, J Li, DF Wong, LS Chao, X Wang…	Self-attention model has shown its flexibility in parallel computation and the effectiveness on modeling both long-and short-term dependencies. However, it calculates the dependencies between representations without considering the contextual information, which has proven …	\N	2019	https://www.aaai.org/Papers/AAAI/2019/AAAI-YangBaosong.4589.pdf	t	15
1337	Improving question answering by commonsense-based pre-training	W Zhong, D Tang, N Duan, M Zhou, J Wang…	Although neural network approaches achieve remarkable success on a variety of NLP tasks, many of them struggle to answer questions that require commonsense knowledge. We believe the main reason is the lack of commonsense connections between concepts. To …	\N	2019	https://link.springer.com/chapter/10.1007/978-3-030-32233-5_2	t	14
2500	Overview of the task on irony detection in spanish variants	R Ortega-Bueno, F Rangel…	This paper introduces IroSvA, the first shared task fully dedicated to identify the presence of irony in short messages (tweets and news comments) written in three different variants of Spanish. The task consists in: given a message, automatic systems should recognize …	\N	2019	https://pdfs.semanticscholar.org/90fc/7fe0e4e21f154919e318e447d57f09f08627.pdf	t	14
1291	Large scale language modeling: Converging on 40gb of text in four hours	R Puri, R Kirby, N Yakovenko…	Recent work has shown how to train Convolutional Neural Networks (CNNs) rapidly on large image datasets [1], then transfer the knowledge gained from these models to a variety of tasks [2]. Following [3], in this work, we demonstrate similar scalability and transfer for …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8645935/	t	14
1287	Subword-augmented embedding for cloze reading comprehension	Z Zhang, Y Huang, H Zhao	Representation learning is the foundation of machine reading comprehension. In state-of-the-art models, deep learning methods broadly use word and character level representations. However, character is not naturally the minimal linguistic unit. In addition …	\N	2018	https://arxiv.org/abs/1806.09103	t	14
1371	Direct Output Connection for a High-Rank Language Model	S Takase, J Suzuki, M Nagata	This paper proposes a state-of-the-art recurrent neural network (RNN) language model that combines probability distributions computed not only from a final RNN layer but also from middle layers. Our proposed method raises the expressive power of a language model …	\N	2018	https://arxiv.org/abs/1808.10143	t	14
1360	Clinical Concept Extraction with Contextual Word Embedding	H Zhu, IC Paschalidis, A Tahmasebi	Automatic extraction of clinical concepts is an essential step for turning the unstructured data within a clinical note into structured and actionable information. In this work, we propose a clinical concept extraction model for automatic annotation of clinical problems, treatments …	\N	2018	https://arxiv.org/abs/1810.10566	t	14
1326	Adventure: Adversarial training for textual entailment with knowledge-guided examples	D Kang, T Khot, A Sabharwal, E Hovy	We consider the problem of learning textual entailment models with limited supervision (5K-10K training examples), and present two complementary approaches for it. First, we propose knowledge-guided adversarial example generators for incorporating large lexical resources …	\N	2018	https://arxiv.org/abs/1805.04680	t	14
1768	A Theoretical Analysis of Contrastive Unsupervised Representation Learning	S Arora, H Khandeparkar, M Khodak…	Recent empirical works have successfully used unlabeled data to learn feature representations that are broadly useful in downstream classification tasks. Several of these methods are reminiscent of the well-known word2vec embedding algorithm: leveraging …	\N	2019	https://arxiv.org/abs/1902.09229	t	14
2503	CEDR: Contextualized embeddings for document ranking	S MacAvaney, A Yates, A Cohan…	Although considerable attention has been given to neural ranking architectures recently, far less attention has been paid to the term representations that are used as input to these models. In this work, we investigate how two pretrained contextualized language models …	\N	2019	https://dl.acm.org/doi/abs/10.1145/3331184.3331317	t	13
2501	CLPsych 2019 shared task: Predicting the degree of suicide risk in Reddit posts	A Zirikly, P Resnik, O Uzuner…	The shared task for the 2019 Workshop on Computational Linguistics and Clinical Psychology (CLPsych'19) introduced an assessment of suicide risk based on social media postings, using data from Reddit to identify users at no, low, moderate, or severe risk. Two …	\N	2019	https://www.aclweb.org/anthology/W19-3003.pdf	t	13
1296	End-task oriented textual entailment via deep explorations of inter-sentence interactions	W Yin, H Schütze, D Roth	This work deals with SciTail, a natural entailment challenge derived from a multi-choice question answering problem. The premises and hypotheses in SciTail were generated with no awareness of each other, and did not specifically aim at the entailment task. This makes it …	\N	2018	https://arxiv.org/abs/1804.08813	t	13
1393	Retrieve-and-read: Multi-task learning of information retrieval and reading comprehension	K Nishida, I Saito, A Otsuka, H Asano…	This study considers the task of machine reading at scale (MRS) wherein, given a question, a system first performs the information retrieval (IR) task of finding relevant passages in a knowledge source and then carries out the reading comprehension (RC) task of extracting …	\N	2018	https://dl.acm.org/citation.cfm?id=3271702	t	13
1317	Hierarchical multitask learning for CTC-based speech recognition	K Krishna, S Toshniwal, K Livescu	Previous work has shown that neural encoder-decoder speech recognition can be improved with hierarchical multitask learning, where auxiliary tasks are added at intermediate layers of a deep encoder. We explore the effect of hierarchical multitask learning in the context of …	\N	2018	https://arxiv.org/abs/1807.06234	t	13
1412	Understanding learning dynamics of language models with svcca	N Saphra, A Lopez	Recent work has demonstrated that neural language models encode linguistic structure implicitly in a number of ways. However, existing research has not shed light on the process by which this structure is acquired during training. We use SVCCA as a tool for …	\N	2018	https://arxiv.org/abs/1811.00225	t	13
1403	A Tutorial on Deep Latent Variable Models of Natural Language	Y Kim, S Wiseman, AM Rush	There has been much recent, exciting work on combining the complementary strengths of latent variable models and deep learning. Latent variable modeling makes it easy to explicitly specify model constraints through conditional independence properties, while …	\N	2018	https://arxiv.org/abs/1812.06834	t	13
1339	A span selection model for semantic role labeling	H Ouchi, H Shindo, Y Matsumoto	We present a simple and accurate span-based model for semantic role labeling (SRL). Our model directly takes into account all possible argument spans and scores them for each label. At decoding time, we greedily select higher scoring labeled spans. One advantage of …	\N	2018	https://arxiv.org/abs/1810.02245	t	13
1463	An Unsupervised Autoregressive Model for Speech Representation Learning	YA Chung, WN Hsu, H Tang, J Glass	This paper proposes a novel unsupervised autoregressive neural model for learning generic speech representations. In contrast to other speech representation learning methods that aim to remove noise or speaker variabilities, ours is designed to preserve information for a …	\N	2019	https://arxiv.org/abs/1904.03240	t	13
2502	Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training	G Li, N Duan, Y Fang, D Jiang, M Zhou	We propose Unicoder-VL, a universal encoder that aims to learn joint representations of vision and language in a pre-training manner. Borrow ideas from cross-lingual pre-trained models, such as XLM and Unicoder, both visual and linguistic contents are fed into a multi …	\N	2019	https://arxiv.org/abs/1908.06066	t	12
2504	DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter	V Sanh, L Debut, J Chaumond, T Wolf	As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP), operating these large models in on-the-edge and/or under constrained computational training or inference budgets remain challenging. In this …	\N	2019	https://arxiv.org/abs/1910.01108	t	12
1730	Dynamic Layer Aggregation for Neural Machine Translation with Routing-by-Agreement	ZY Dou, Z Tu, X Wang, L Wang, S Shi…	With the promising progress of deep neural networks, layer aggregation has been used to fuse information across layers in various fields, such as computer vision and machine translation. However, most of the previous methods combine layers in a static fashion in that …	\N	2019	https://www.aaai.org/Papers/AAAI/2019/AAAI-DouZ.3868.pdf	t	12
1380	An Embarrassingly Simple Approach for Transfer Learning from Pretrained Language Models	A Chronopoulou, C Baziotis, A Potamianos	A growing number of state-of-the-art transfer learning methods employ language models pretrained on large generic corpora. In this paper we present a conceptually simple and effective transfer learning approach that addresses the problem of catastrophic forgetting …	\N	2019	https://arxiv.org/abs/1902.10547	t	12
1340	Deriving machine attention from human rationales	Y Bao, S Chang, M Yu, R Barzilay	Attention-based models are successful when trained on large amounts of data. In this paper, we demonstrate that even in the low-resource scenario, attention can be learned effectively. To this end, we start with discrete human-annotated rationales and map them into …	\N	2018	https://arxiv.org/abs/1808.09367	t	12
1329	Proppy: Organizing the news based on their propagandistic content	A Barrón-Cedeno, I Jaradat, G Da San Martino…	Propaganda is a mechanism to influence public opinion, which is inherently present in extremely biased and fake news. Here, we propose a model to automatically assess the level of propagandistic content in an article based on different representations, from writing …	\N	2019	https://www.sciencedirect.com/science/article/pii/S0306457318306058	t	12
2505	Casm: A deep-learning approach for identifying collective action events with text and image data from social media	H Zhang, J Pan	Protest event analysis is an important method for the study of collective action and social movements and typically draws on traditional media reports as the data source. We introduce collective action from social media (CASM)—a system that uses convolutional …	\N	2019	https://journals.sagepub.com/doi/abs/10.1177/0081175019860244	t	11
2506	Transfer learning in natural language processing	S Ruder, ME Peters, S Swayamdipta…	The classic supervised machine learning paradigm is based on learning in isolation, a single predictive model for a task using a single dataset. This approach requires a large number of training examples and performs best for well-defined and narrow tasks. Transfer …	\N	2019	https://www.aclweb.org/anthology/N19-5004.pdf	t	11
1315	Zero-Shot Cross-lingual Classification Using Multilingual Neural Machine Translation	A Eriguchi, M Johnson, O Firat, H Kazawa…	Transferring representations from large supervised tasks to downstream tasks has shown promising results in AI fields such as Computer Vision and Natural Language Processing (NLP). In parallel, the recent progress in Machine Translation (MT) has enabled one to train …	\N	2018	https://arxiv.org/abs/1809.04686	t	11
1375	Unsupervised Latent Tree Induction with Deep Inside-Outside Recursive Autoencoders	A Drozdov, P Verga, M Yadav, M Iyyer…	We introduce deep inside-outside recursive autoencoders (DIORA), a fully-unsupervised method for discovering syntax that simultaneously learns representations for constituents within the induced tree. Our approach predicts each word in an input sentence conditioned …	\N	2019	https://arxiv.org/abs/1904.02142	t	11
1313	Modeling language variation and universals: A survey on typological linguistics for natural language processing	EM Ponti, H O'Horan, Y Berzak, I Vulić…	Addressing the cross-lingual variation of grammatical structures and meaning categorization is a key challenge for multilingual Natural Language Processing. The lack of resources for the majority of the world's languages makes supervised learning not viable. Moreover, the …	\N	2018	https://www.mitpressjournals.org/doi/abs/10.1162/COLI_a_00357	t	11
1624	Information Aggregation for Multi-Head Attention with Routing-by-Agreement	J Li, B Yang, ZY Dou, X Wang, MR Lyu, Z Tu	Multi-head attention is appealing for its ability to jointly extract different types of information from multiple representation subspaces. Concerning the information aggregation, a common practice is to use a concatenation followed by a linear transformation, which may not fully …	\N	2019	https://arxiv.org/abs/1904.03100	t	11
2507	Wic: the word-in-context dataset for evaluating context-sensitive meaning representations	MT Pilehvar, J Camacho-Collados	By design, word embeddings are unable to model the dynamic nature of words' semantics, ie, the property of words to correspond to potentially different meanings. To address this limitation, dozens of specialized meaning representation techniques such as sense or …	\N	2019	https://www.aclweb.org/anthology/N19-1128.pdf	t	10
1343	Differentiable Perturb-and-Parse: Semi-Supervised Parsing with a Structured Variational Autoencoder	C Corro, I Titov	Human annotation for syntactic parsing is expensive, and large resources are available only for a fraction of languages. A question we ask is whether one can leverage abundant unlabeled texts to improve syntactic parsers, beyond just using the texts to obtain more …	\N	2018	https://arxiv.org/abs/1807.09875	t	10
1325	On the Relation between Linguistic Typology and (Limitations of) Multilingual Language Modeling	D Gerz, I Vulić, EM Ponti, R Reichart…	A key challenge in cross-lingual NLP is developing general language-independent architectures that are equally applicable to any language. However, this ambition is largely hampered by the variation in structural and semantic properties, ie the typological profiles of …	\N	2018	https://www.aclweb.org/anthology/papers/D/D18/D18-1029/	t	10
1723	Learning protein sequence embeddings using information from structure	T Bepler, B Berger	Inferring the structural properties of a protein from its amino acid sequence is a challenging yet important problem in biology. Structures are not known for the vast majority of protein sequences, but structure is critical for understanding function. Existing approaches for …	\N	2019	https://arxiv.org/abs/1902.08661	t	10
1363	Efficient contextualized representation: Language model pruning for sequence labeling	L Liu, X Ren, J Shang, J Peng, J Han	Many efforts have been made to facilitate natural language processing tasks with pre-trained language models (PTLM), and brought significant improvements to various applications. To fully leverage the nearly unlimited corpora and capture linguistic information of multifarious …	\N	2018	https://arxiv.org/abs/1804.07827	t	10
1365	Revisiting LSTM Networks for Semi-Supervised Text Classification via Mixed Objective Function	DS Sachan, M Zaheer…	In this paper, we study bidirectional LSTM network for the task of text classification using both supervised and semi-supervised approaches. Several prior works have suggested that either complex pretraining schemes using unsupervised methods such as language …	\N	2019	https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4672	t	9
2508	Patient knowledge distillation for bert model compression	S Sun, Y Cheng, Z Gan, J Liu	Pre-trained language models such as BERT have proven to be highly effective for natural language processing (NLP) tasks. However, the high demand for computing resources in training such models hinders their application in practice. In order to alleviate this resource …	\N	2019	https://arxiv.org/abs/1908.09355	t	9
1331	Illustrative Language Understanding: Large-Scale Visual Grounding with Image Search	J Kiros, W Chan, G Hinton	We introduce Picturebook, a large-scale lookup operation to ground language via 'snapshots' of our physical world accessed through image search. For each word in a vocabulary, we extract the top-k images from Google image search and feed the images …	\N	2018	https://www.aclweb.org/anthology/P18-1085.pdf	t	9
2509	Multilingual extractive reading comprehension by runtime machine translation	A Asai, A Eriguchi, K Hashimoto, Y Tsuruoka	Despite recent work in Reading Comprehension (RC), progress has been mostly limited to English due to the lack of large-scale datasets in other languages. In this work, we introduce the first RC system for languages without RC training data. Given a target language without …	\N	2018	https://arxiv.org/abs/1809.03275	t	9
1345	A helping hand: Transfer learning for deep sentiment analysis	XL Dong, G De Melo	Deep convolutional neural networks excel at sentiment polarity classification, but tend to require substantial amounts of training data, which moreover differs quite significantly between domains. In this work, we present an approach to feed generic cues into the …	\N	2018	https://www.aclweb.org/anthology/P18-1235.pdf	t	9
2510	Uniter: Learning universal image-text representations	YC Chen, L Li, L Yu, AE Kholy, F Ahmed, Z Gan…	Joint image-text embedding is the bedrock for most Vision-and-Language (V+ L) tasks, where multimodality inputs are jointly processed for visual and textual understanding. In this paper, we introduce UNITER, a UNiversal Image-TExt Representation, learned through …	\N	2019	https://arxiv.org/abs/1909.11740	t	9
1328	Extending a parser to distant domains using a few dozen partially annotated examples	V Joshi, M Peters, M Hopkins	We revisit domain adaptation for parsers in the neural era. First we show that recent advances in word representations greatly diminish the need for domain adaptation when the target domain is syntactically similar to the source domain. As evidence, we train a parser on …	\N	2018	https://arxiv.org/abs/1805.06556	t	9
1610	Constituent Parsing as Sequence Labeling	C Gómez-Rodríguez, D Vilares	We introduce a method to reduce constituent parsing to sequence labeling. For each word w_t, it generates a label that encodes:(1) the number of ancestors in the tree that the words w_t and w_ {t+ 1} have in common, and (2) the nonterminal symbol at the lowest common …	\N	2018	https://arxiv.org/abs/1810.08994	t	9
1392	Simple and effective semi-supervised question answering	B Dhingra, D Pruthi, D Rajagopal	Recent success of deep learning models for the task of extractive Question Answering (QA) is hinged on the availability of large annotated corpora. However, large domain specific annotated corpora are limited and expensive to construct. In this work, we envision a system …	\N	2018	https://arxiv.org/abs/1804.00720	t	9
1409	Unsupervised Learning of Syntactic Structure with Invertible Neural Projections	J He, G Neubig, T Berg-Kirkpatrick	Unsupervised learning of syntactic structure is typically performed using generative models with discrete latent variables and multinomial parameters. In most cases, these models have not leveraged continuous word representations. In this work, we propose a novel generative …	\N	2018	https://arxiv.org/abs/1808.09111	t	9
1373	What's in your embedding, and how it predicts task performance	A Rogers, SH Ananthakrishna…	Attempts to find a single technique for general-purpose intrinsic evaluation of word embeddings have so far not been successful. We present a new approach based on scaled-up qualitative analysis of word vector neighborhoods that quantifies interpretable …	\N	2018	https://www.aclweb.org/anthology/C18-1228.pdf	t	8
2511	Universal adversarial triggers for nlp	E Wallace, S Feng, N Kandpal, M Gardner…	Adversarial examples highlight model vulnerabilities and are useful for evaluation and interpretation. We define universal adversarial triggers: input-agnostic sequences of tokens that trigger a model to produce a specific prediction when concatenated to any input from a …	\N	2019	https://arxiv.org/abs/1908.07125	t	8
1404	Rethinking Complex Neural Network Architectures for Document Classification	A Adhikari, A Ram, R Tang, J Lin	Neural network models for many NLP tasks have grown increasingly complex in recent years, making training and deployment more difficult. A number of recent papers have questioned the necessity of such architectures and found that well-executed, simpler models …	\N	2019	https://www.aclweb.org/anthology/N19-1408.pdf	t	8
2512	On difficulties of cross-lingual transfer with order differences: A case study on dependency parsing	W Ahmad, Z Zhang, X Ma, E Hovy, KW Chang…	Different languages might have different word orders. In this paper, we investigate crosslingual transfer and posit that an orderagnostic model will perform better when transferring to distant foreign languages. To test our hypothesis, we train dependency …	\N	2019	https://www.aclweb.org/anthology/N19-1253.pdf	t	8
2513	Overview of the ntcir-14 finnum task: Fine-grained numeral understanding in financial social media data	CC Chen, HH Huang, H Takamura…	Numeral is the crucial part of financial documents. In order to understand the detail of opinions in financial documents, we should not only analyze the text, but also need to assay the numeric information in depth. Because of the informal writing style, analyzing social …	\N	2019	https://www.cs.nccu.edu.tw/~hhhuang/docs/ntcir2019.pdf	t	8
599	A comprehensive survey of graph embedding: Problems, techniques, and applications	H Cai, VW Zheng, KCC Chang	Graph is an important data representation which appears in a wide diversity of real-world scenarios. Effective graph analytics provides users a deeper understanding of what is behind the data, and thus can benefit a lot of useful applications such as node classification …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8294302/	t	353
587	Boosting image captioning with attributes	T Yao, Y Pan, Y Li, Z Qiu, T Mei	Automatically describing an image with a natural language has been an emerging challenge in both fields of computer vision and natural language processing. In this paper, we present Long Short-Term Memory with Attributes (LSTM-A)-a novel architecture that …	\N	2017	http://openaccess.thecvf.com/content_iccv_2017/html/Yao_Boosting_Image_Captioning_ICCV_2017_paper.html	t	266
2137	Transfer learning for sequence tagging with hierarchical recurrent networks	Z Yang, R Salakhutdinov, WW Cohen	Recent papers have shown that neural networks obtain state-of-the-art performance on several different sequence tagging tasks. One appealing property of such systems is their generality, as excellent performance can be achieved with a unified architecture and without …	\N	2017	https://arxiv.org/abs/1703.06345	t	141
2135	Inducing domain-specific sentiment lexicons from unlabeled corpora	WL Hamilton, K Clark, J Leskovec…	A word's sentiment depends on the domain in which it is used. Computational social science research thus requires sentiment lexicons that are specific to the domains being studied. We combine domain-specific word embeddings with a label propagation framework to induce …	\N	2016	https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5483533/	t	139
2144	GRAM: graph-based attention model for healthcare representation learning	E Choi, MT Bahadori, L Song, WF Stewart…	Deep learning methods exhibit promising performance for predic- tive modeling \nin healthcare, but two important challenges remain: • Data insu ciency: Often in healthcare predictive \nmodeling, the sample size is insu cient for deep learning methods to achieve satisfactory …	\N	2017	https://dl.acm.org/citation.cfm?id=3098126	t	139
2125	Counter-fitting word vectors to linguistic constraints	N Mrkšić, DO Séaghdha, B Thomson, M Gašić…	In this work, we present a novel counter-fitting method which injects antonymy and synonymy constraints into vector space representations in order to improve the vectors' capability for judging semantic similarity. Applying this method to publicly available pre …	\N	2016	https://arxiv.org/abs/1603.00892	t	137
2126	Stochastic optimization for large-scale optimal transport	A Genevay, M Cuturi, G Peyré, F Bach	Optimal transport (OT) defines a powerful framework to compare probability distributions in a geometrically faithful way. However, the practical impact of OT is still limited because of its computational burden. We propose a new class of stochastic optimization algorithms to cope …	\N	2016	http://papers.nips.cc/paper/6566-stochastic-optimization-for-large-scale-optimal-transport	t	137
669	De-identification of patient notes with recurrent neural networks	F Dernoncourt, JY Lee, O Uzuner…	Objective: Patient notes in electronic health records (EHRs) may contain critical information for medical investigations. However, the vast majority of medical investigators can only access de-identified notes, in order to protect the confidentiality of patients. In the United …	\N	2017	https://academic.oup.com/jamia/article-abstract/24/3/596/2769353	t	137
698	Understanding neural networks through representation erasure	J Li, W Monroe, D Jurafsky	While neural networks have been successfully applied to many natural language processing tasks, they come at the cost of interpretability. In this paper, we propose a general methodology to analyze and interpret decisions from a neural model by observing the effects …	\N	2016	https://arxiv.org/abs/1612.08220	t	137
2174	Personalizing Dialogue Agents: I have a dog, do you have pets too?	S Zhang, E Dinan, J Urbanek, A Szlam, D Kiela…	Chit-chat models are known to have several problems: they lack specificity, do not display a consistent personality and are often not very captivating. In this work we present the task of making chit-chat more engaging by conditioning on profile information. We collect data and …	\N	2018	https://arxiv.org/abs/1801.07243	t	136
662	Semantic similarity from natural language and ontology analysis	S Harispe, S Ranwez, S Janaqi…	Artificial Intelligence federates numerous scientific fields in the aim of developing machines able to assist human operators performing complex treatments---most of which demand high cognitive skills (eg learning or decision processes). Central to this quest is to give machines …	\N	2015	https://www.morganclaypool.com/doi/abs/10.2200/S00639ED1V01Y201504HLT027	t	135
2129	Tips and tricks for visual question answering: Learnings from the 2017 challenge	D Teney, P Anderson, X He…	This paper presents a state-of-the-art model for visual question answering (VQA), which won the first place in the 2017 VQA Challenge. VQA is a task of significant importance for research in artificial intelligence, given its multimodal nature, clear evaluation protocol, and …	\N	2018	http://openaccess.thecvf.com/content_cvpr_2018/html/Teney_Tips_and_Tricks_CVPR_2018_paper.html	t	135
642	Studying user income through language, behaviour and affect in social media	D Preoţiuc-Pietro, S Volkova, V Lampos, Y Bachrach…	Automatically inferring user demographics from social media posts is useful for both social science research and a range of downstream applications in marketing and politics. We present the first extensive study where user behaviour on Twitter is used to build a predictive …	\N	2015	https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0138717	t	134
670	Sentiment embeddings with applications to sentiment analysis	D Tang, F Wei, B Qin, N Yang, T Liu…	We propose learning sentiment-specific word embeddings dubbed sentiment embeddings in this paper. Existing word embedding learning algorithms typically only use the contexts of words but ignore the sentiment of texts. It is problematic for sentiment analysis because the …	\N	2015	https://ieeexplore.ieee.org/abstract/document/7296633/	t	134
2114	Computational linguistics and deep learning	CD Manning	Deep Learning waves have lapped at the shores of computational linguistics for several years now, but 2015 seems like the year when the full force of the tsunami hit the major Natural Language Processing (NLP) conferences. However, some pundits are predicting …	\N	2015	https://www.mitpressjournals.org/doi/full/10.1162/COLI_a_00239	t	133
2138	Neural ranking models with weak supervision	M Dehghani, H Zamani, A Severyn, J Kamps…	Despite the impressive improvements achieved by unsupervised deep neural networks in computer vision and NLP tasks, such improvements have not yet been observed in ranking for information retrieval. The reason may be the complexity of the ranking problem, as it is …	\N	2017	https://dl.acm.org/citation.cfm?id=3080832	t	133
602	Classification of sentiment reviews using n-gram machine learning approach	A Tripathy, A Agrawal, SK Rath	With the ever increasing social networking and online marketing sites, the reviews and blogs obtained from those, act as an important source for further analysis and improved decision making. These reviews are mostly unstructured by nature and thus, need processing like …	\N	2016	https://www.sciencedirect.com/science/article/pii/S095741741630118X	t	224
607	Representation learning of knowledge graphs with entity descriptions	R Xie, Z Liu, J Jia, H Luan, M Sun	Representation learning (RL) of knowledge graphs aims to project both entities and relations into a continuous low-dimensional space. Most methods concentrate on learning representations with knowledge triples indicating relations between entities. In fact, in most …	\N	2016	https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12216	t	202
2119	Pairwise word interaction modeling with deep neural networks for semantic similarity measurement	H He, J Lin	Textual similarity measurement is a challenging problem, as it requires understanding the semantics of input sentences. Most previous neural network models use coarse-grained sentence modeling, which has difficulty capturing fine-grained word-level information for …	\N	2016	https://www.aclweb.org/anthology/N16-1108	t	132
677	Topic modeling for short texts with auxiliary word embeddings	C Li, H Wang, Z Zhang, A Sun, Z Ma	For many applications that require semantic understanding of short texts, inferring discriminative and coherent latent topics from short texts is a critical and fundamental task. Conventional topic models largely rely on word co-occurrences to derive topics from a …	\N	2016	https://dl.acm.org/citation.cfm?id=2911499	t	130
2142	From softmax to sparsemax: A sparse model of attention and multi-label classification	A Martins, R Astudillo	We propose sparsemax, a new activation function similar to the traditional softmax, but able to output sparse probabilities. After deriving its properties, we show how its Jacobian can be efficiently computed, enabling its use in a network trained with backpropagation. Then, we …	\N	2016	http://www.jmlr.org/proceedings/papers/v48/martins16.pdf	t	130
2154	Lifelong machine learning	Z Chen, B Liu	NOTE⁃ A New Edition of This Title is Available: Lifelong Machine Learning, Second Edition Lifelong Machine Learning (or Lifelong Learning) is an advanced machine learning paradigm that learns continuously, accumulates the knowledge learned in previous tasks …	\N	2016	https://www.morganclaypool.com/doi/abs/10.2200/S00737ED1V01Y201610AIM033	t	128
2141	A compare-aggregate model for matching text sequences	S Wang, J Jiang	Many NLP tasks including machine comprehension, answer selection and text entailment require the comparison between sequences. Matching the important units between sequences is a key to solve these problems. In this paper, we present a general" compare …	\N	2016	https://arxiv.org/abs/1611.01747	t	127
2122	SemEval-2017 task 3: Community question answering	P Nakov, D Hoogeveen, L Màrquez, A Moschitti…	We describe SemEval–2017 Task 3 on Community Question Answering. This year, we reran the four subtasks from SemEval-2016:(A) Question–Comment Similarity,(B) Question–Question Similarity,(C) Question–External Comment Similarity, and (D) Rerank …	\N	2019	https://arxiv.org/abs/1912.00730	t	126
2175	Simple and effective multi-paragraph reading comprehension	C Clark, M Gardner	We consider the problem of adapting neural paragraph-level question answering models to the case where entire documents are given as input. Our proposed solution trains models to produce well calibrated confidence scores for their results on individual paragraphs. We …	\N	2017	https://arxiv.org/abs/1710.10723	t	124
2157	Learning to represent programs with graphs	M Allamanis, M Brockschmidt, M Khademi	Learning tasks on source code (ie, formal languages) have been considered recently, but most work has tried to transfer natural language methods and does not capitalize on the unique opportunities offered by code's known syntax. For example, long-range …	\N	2017	https://arxiv.org/abs/1711.00740	t	123
2158	Modulating early visual processing by language	H De Vries, F Strub, J Mary, H Larochelle…	It is commonly assumed that language refers to high-level visual concepts while leaving low-level visual processing unaffected. This view dominates the current literature in computational models for language-vision tasks, where visual and linguistic inputs are …	\N	2017	http://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language	t	122
2151	Unsupervised pretraining for sequence to sequence learning	P Ramachandran, PJ Liu, QV Le	This work presents a general unsupervised learning method to improve the accuracy of sequence to sequence (seq2seq) models. In our method, the weights of the encoder and decoder of a seq2seq model are initialized with the pretrained weights of two language …	\N	2016	https://arxiv.org/abs/1611.02683	t	121
2128	User modeling with neural network for review rating prediction	D Tang, B Qin, T Liu, Y Yang	We present a neural network method for review rating prediction in this paper. Existing neural network methods for sentiment prediction typically only capture the semantics of texts, but ignore the user who expresses the sentiment. This is not desirable for review rating …	\N	2015	https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/viewPaper/11051	t	118
2170	Empower sequence labeling with task-aware neural language model	L Liu, J Shang, X Ren, FF Xu, H Gui, J Peng…	Linguistic sequence labeling is a general approach encompassing a variety of problems, such as part-of-speech tagging and named entity recognition. Recent advances in neural networks (NNs) make it possible to build reliable models without handcrafted features …	\N	2018	https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17123	t	118
2139	Beyond binary labels: political ideology prediction of twitter users	D Preoţiuc-Pietro, Y Liu, D Hopkins…	Automatic political orientation prediction from social media posts has to date proven successful only in distinguishing between publicly declared liberals and conservatives in the US. This study examines users' political ideology using a seven-point scale which enables …	\N	2017	https://www.aclweb.org/anthology/papers/P/P17/P17-1068/	t	114
2171	Learning general purpose distributed sentence representations via large scale multi-task learning	S Subramanian, A Trischler, Y Bengio…	A lot of the recent success in natural language processing (NLP) has been driven by distributed vector representations of words trained on large amounts of text in an unsupervised manner. These representations are typically used as general purpose …	\N	2018	https://arxiv.org/abs/1804.00079	t	114
2166	Attributed social network embedding	L Liao, X He, H Zhang, TS Chua	Embedding network data into a low-dimensional vector space has shown promising performance for many real-world applications, such as node classification and entity retrieval. However, most existing methods focused only on leveraging network structure. For …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8326519/	t	112
1323	Multimodal relational tensor network for sentiment and emotion classification	S Sahay, SH Kumar, R Xia, J Huang…	Understanding Affect from video segments has brought researchers from the language, audio and video domains together. Most of the current multimodal research in this area deals with various techniques to fuse the modalities, and mostly treat the segments of a …	\N	2018	https://arxiv.org/abs/1806.02923	t	8
1314	Wic: 10,000 example pairs for evaluating context-sensitive representations	MT Pilehvar, J Camacho-Collados	By design, word embeddings are unable to model the dynamic nature of words' semantics, ie, the property of words to correspond to potentially different meanings depending on the context in which they appear. To address this limitation, dozens of specialized word …	\N	2018	https://arxiv.org/abs/1808.09121	t	8
1352	Don't Settle for Average, Go for the Max: Fuzzy Sets and Max-Pooled Word Vectors	V Zhelezniak, A Savkov, A Shen, F Moramarco…	Recent literature suggests that averaged word vectors followed by simple post-processing outperform many deep learning methods on semantic textual similarity tasks. Furthermore, when averaged word vectors are trained supervised on large corpora of paraphrases, they …	\N	2019	https://arxiv.org/abs/1904.13264	t	8
1429	Generative Question Answering: Learning to Answer the Whole Question	M Lewis, A Fan	Discriminative question answering models can overfit to superficial biases in datasets, because their loss function saturates when any clue makes the answer likely. We introduce generative models of the joint distribution of questions and answers, which are trained to …	\N	2018	https://openreview.net/forum?id=Bkx0RjA9tX	t	8
2516	Text summarization with pretrained encoders	Y Liu, M Lapata	Bidirectional Encoder Representations from Transformers (BERT) represents the latest incarnation of pretrained language models which have recently advanced a wide range of natural language processing tasks. In this paper, we showcase how BERT can be usefully …	\N	2019	https://arxiv.org/abs/1908.08345	t	7
2514	SMIT: Stochastic multi-label image-to-image translation	A Romero, P Arbeláez, L Van Gool…	Cross-domain mapping has been a very active topic in recent years. Given one image, its main purpose is to translate it to the desired target domain, or multiple domains in the case of multiple labels. This problem is highly challenging due to three main reasons:(i) unpaired …	\N	2019	http://openaccess.thecvf.com/content_ICCVW_2019/html/AIM/Romero_SMIT_Stochastic_Multi-Label_Image-to-Image_Translation_ICCVW_2019_paper.html	t	7
1689	Words Can Shift: Dynamically Adjusting Word Representations Using Nonverbal Behaviors	Y Wang, Y Shen, Z Liu, PP Liang, A Zadeh…	Humans convey their intentions through the usage of both verbal and nonverbal behaviors during face-to-face communication. Speaker intentions often vary dynamically depending on different nonverbal contexts, such as vocal patterns and facial expressions. As a result …	\N	2019	https://www.aaai.org/ojs/index.php/AAAI/article/view/4706	t	7
1320	Attention-guided answer distillation for machine reading comprehension	M Hu, Y Peng, F Wei, Z Huang, D Li, N Yang…	Despite that current reading comprehension systems have achieved significant advancements, their promising performances are often obtained at the cost of making an ensemble of numerous models. Besides, existing approaches are also vulnerable to …	\N	2018	https://arxiv.org/abs/1808.07644	t	7
1357	Verb argument structure alternations in word and sentence embeddings	K Kann, A Warstadt, A Williams, SR Bowman	Verbs occur in different syntactic environments, or frames. We investigate whether artificial neural networks encode grammatical distinctions necessary for inferring the idiosyncratic frame-selectional properties of verbs. We introduce five datasets, collectively called FAVA …	\N	2018	https://arxiv.org/abs/1811.10773	t	7
1359	Inline Detection of Domain Generation Algorithms with Context-Sensitive Word Embeddings	JJ Koh, B Rhodes	Domain generation algorithms (DGAs) are frequently employed by malware to generate domains used for connecting to command-and-control (C2) servers. Recent work in DGA detection leveraged deep learning architectures like convolutional neural networks (CNNs) …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8622066/	t	7
2521	Wide-coverage neural A* parsing for minimalist grammars	J Torr, M Stanojevic, M Steedman…	Minimalist Grammars (Stabler, 1997) are a computationally oriented, and rigorous formalisation of many aspects of Chomsky's (1995) Minimalist Program. This paper presents the first ever application of this formalism to the task of realistic wide-coverage parsing. The …	\N	2019	https://www.aclweb.org/anthology/P19-1238.pdf	t	6
2515	Leveraging pre-trained checkpoints for sequence generation tasks	S Rothe, S Narayan, A Severyn	Unsupervised pre-training of large neural models has recently revolutionized Natural Language Processing. Warm-starting from the publicly released checkpoints, NLP practitioners have pushed the state-of-the-art on multiple benchmarks while saving …	\N	2019	https://arxiv.org/abs/1907.12461	t	6
2517	Megatron-lm: Training multi-billion parameter language models using gpu model parallelism	M Shoeybi, M Patwary, R Puri, P LeGresley…	Recent work in unsupervised language modeling demonstrates that training large neural language models advances the state of the art in Natural Language Processing applications. However, for very large models, memory constraints limit the size of models …	\N	2019	https://arxiv.org/abs/1909.08053	t	6
1335	Spell once, summon anywhere: A two-level open-vocabulary language model	SJ Mielke, J Eisner	We show how to deploy recurrent neural networks within a hierarchical Bayesian language model. Our generative story combines a standard RNN language model (generating the word tokens in each sentence) with an RNN-based spelling model (generating the letters in …	\N	2019	https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4660	t	6
2518	Can you tell me how to get past sesame street? sentence-level pretraining beyond language modeling	A Wang, J Hula, P Xia, R Pappagari…	Natural language understanding has recently seen a surge of progress with the use of sentence encoders like ELMo (Peters et al., 2018a) and BERT (Devlin et al., 2019) which are pretrained on variants of language modeling. We conduct the first large-scale systematic …	\N	2019	https://www.aclweb.org/anthology/P19-1439.pdf	t	6
1379	InferLite: Simple Universal Sentence Representations from Natural Language Inference Data	J Kiros, W Chan	Natural language inference has been shown to be an effective supervised task for learning generic sentence embeddings. In order to better understand the components that lead to effective representations, we propose a lightweight version of InferSent, called InferLite, that …	\N	2018	https://www.aclweb.org/anthology/D18-1524.pdf	t	6
1638	DSTC7 Task 1: Noetic End-to-End Response Selection	C Gunasekara, JK Kummerfeld…	Goal-oriented dialogue in complex domains is an extremely challenging problem and there are relatively few datasets. This task provided two new resources that presented different challenges: one was focused but small, while the other was large but diverse. We also …	\N	2019	https://www.aclweb.org/anthology/W19-4107.pdf	t	6
1415	Pythia-a platform for vision & language research	A Singh, V Natarajan, Y Jiang…	This paper presents Pythia, a deep learning research platform for vision & language tasks. Pythia is built with a plug-&-play strategy at its core, which enables researchers to quickly build, reproduce and benchmark novel models for vision & language tasks like Visual …	\N	2018	https://pdfs.semanticscholar.org/d6de/df6d25df2a5cd727a019b613953afc9a0300.pdf	t	6
2519	Ernie 2.0: A continual pre-training framework for language understanding	Y Sun, S Wang, Y Li, S Feng, H Tian, H Wu…	Recently, pre-trained models have achieved state-of-the-art results in various language understanding tasks, which indicates that pre-training on large-scale corpora may play a crucial role in natural language processing. Current pre-training procedures usually focus …	\N	2019	https://arxiv.org/abs/1907.12412	t	6
2700	NormCo: Deep disease normalization for biomedical knowledge base construction	D Wright	Biomedical knowledge bases are crucial in modern data-driven biomedical sciences, but automated biomedical knowledge base construction remains challenging. In this paper, we consider the problem of disease entity normalization, an essential task in constructing a …	\N	2019	https://escholarship.org/uc/item/3410q7zk	t	6
1690	Unsupervised Transfer Learning for Spoken Language Understanding in Intelligent Agents	A Siddhant, A Goyal, A Metallinou	User interaction with voice-powered agents generates large amounts of unlabeled utterances. In this paper, we explore techniques to efficiently transfer the knowledge from these unlabeled utterances to improve model performance on Spoken Language …	\N	2019	https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4426	t	6
2520	Cross-domain modeling of sentence-level evidence for document retrieval	ZA Yilmaz, W Yang, H Zhang, J Lin	This paper applies BERT to ad hoc document retrieval on news articles, which requires addressing two challenges: relevance judgments in existing test collections are typically provided only at the document level, and documents often exceed the length that BERT was …	\N	2019	https://www.aclweb.org/anthology/D19-1352/	t	6
1319	Happy together: Learning and understanding appraisal from natural language	A Rajendran, C Zhang, M Abdul-Mageed	In this paper, we explore various approaches for learning two types of appraisal components from happy language. We focus on 'agency'of the author and the 'sociality'involved in happy moments based on the HappyDB dataset. We develop models based on deep neural …	\N	2019	https://arxiv.org/abs/1906.03677	t	6
1779	SemEval 2019 Task 1: Cross-lingual Semantic Parsing with UCCA	D Hershcovich, Z Aizenbud, L Choshen…	We present the SemEval 2019 shared task on UCCA parsing in English, German and French, and discuss the participating systems and results. UCCA is a cross-linguistically applicable framework for semantic representation, which builds on extensive typological …	\N	2019	https://arxiv.org/abs/1903.02953	t	6
1428	Extracting Multiple-Relations in One-Pass with Pre-Trained Transformers	H Wang, M Tan, M Yu, S Chang, D Wang, K Xu…	Most approaches to extraction multiple relations from a paragraph require multiple passes over the paragraph. In practice, multiple passes are computationally expensive and this makes difficult to scale to longer paragraphs and larger text corpora. In this work, we focus …	\N	2019	https://arxiv.org/abs/1902.01030	t	6
1368	IIIDYT at IEST 2018: Implicit Emotion Classification With Deep Contextualized Word Representations	JA Balazs, E Marrese-Taylor, Y Matsuo	In this paper we describe our system designed for the WASSA 2018 Implicit Emotion Shared Task (IEST), which obtained 2$^{\\text {nd}} $ place out of 26 teams with a test macro F1 score of $0.710 $. The system is composed of a single pre-trained ELMo layer for encoding …	\N	2018	https://arxiv.org/abs/1808.08672	t	6
2529	Ctrl: A conditional transformer language model for controllable generation	NS Keskar, B McCann, LR Varshney, C Xiong…	Large-scale language models show promising text generation capabilities, but users cannot easily control particular aspects of the generated text. We release CTRL, a 1.6 billion-parameter conditional transformer language model, trained to condition on control codes …	\N	2019	https://arxiv.org/abs/1909.05858	t	5
2522	Dual adversarial neural transfer for low-resource named entity recognition	JT Zhou, H Zhang, D Jin, H Zhu, M Fang…	We propose a new neural transfer method termed Dual Adversarial Transfer Network (DATNet) for addressing low-resource Named Entity Recognition (NER). Specifically, two variants of DATNet, ie, DATNet-F and DATNet-P, are investigated to explore effective feature …	\N	2019	https://www.aclweb.org/anthology/P19-1336.pdf	t	5
1441	A Multi-Stage Memory Augmented Neural Network for Machine Reading Comprehension	S Yu, SR Indurthi, S Back, H Lee	Reading Comprehension (RC) of text is one of the fundamental tasks in natural language processing. In recent years, several end-to-end neural network models have been proposed to solve RC tasks. However, most of these models suffer in reasoning over long documents …	\N	2018	https://www.aclweb.org/anthology/W18-2603.pdf	t	5
2523	Neural architectures for nested NER through linearization	J Straková, M Straka, J Hajič	We propose two neural network architectures for nested named entity recognition (NER), a setting in which named entities may overlap and also be labeled with more than one label. We encode the nested labels using a linearized scheme. In our first proposed approach, the …	\N	2019	https://arxiv.org/abs/1908.06926	t	5
2524	Abductive commonsense reasoning	C Bhagavatula, RL Bras, C Malaviya…	Abductive reasoning is inference to the most plausible explanation. For example, if Jenny finds her house in a mess when she returns from work, and remembers that she left a window open, she can hypothesize that a thief broke into her house and caused the mess …	\N	2019	https://arxiv.org/abs/1908.05739	t	5
1562	Robust Word Vectors: Context-Informed Embeddings for Noisy Texts	V Malykh, V Logacheva, T Khakhulin	We suggest a new language-independent architecture of robust word vectors (RoVe). It is designed to alleviate the issue of typos, which are common in almost any user-generated content, and hinder automatic text processing. Our model is morphologically motivated …	\N	2018	https://www.aclweb.org/anthology/W18-6108.pdf	t	5
2525	On mutual information maximization for representation learning	M Tschannen, J Djolonga, PK Rubenstein…	Many recent methods for unsupervised or self-supervised representation learning train feature extractors by maximizing an estimate of the mutual information (MI) between different views of the data. This comes with several immediate problems: For example, MI is …	\N	2019	https://arxiv.org/abs/1907.13625	t	5
2526	Dialog state tracking: A neural reading comprehension approach	S Gao, A Sethi, S Aggarwal, T Chung…	Dialog state tracking is used to estimate the current belief state of a dialog given all the preceding conversation. Machine reading comprehension, on the other hand, focuses on building systems that read passages of text and answer questions that require some …	\N	2019	https://arxiv.org/abs/1908.01946	t	5
2534	Bam! born-again multi-task networks for natural language understanding	K Clark, MT Luong, U Khandelwal, CD Manning…	It can be challenging to train multi-task neural networks that outperform or even match their single-task counterparts. To help address this, we propose using knowledge distillation where single-task models teach a multi-task model. We enhance this training with teacher …	\N	2019	https://arxiv.org/abs/1907.04829	t	5
2527	Large-scale transfer learning for natural language generation	S Golovanov, R Kurbanov, S Nikolenko…	Large-scale pretrained language models define state of the art in natural language processing, achieving outstanding performance on a variety of tasks. We study how these architectures can be applied and adapted for natural language generation, comparing a …	\N	2019	https://www.aclweb.org/anthology/P19-1608.pdf	t	5
1743	QuaRel: A Dataset and Models for Answering Questions about Qualitative Relationships	O Tafjord, P Clark, M Gardner, W Yih…	Many natural language questions require recognizing and reasoning with qualitative relationships (eg, in science, economics, and medicine), but are challenging to answer with corpus-based methods. Qualitative modeling provides tools that support such reasoning, but …	\N	2019	https://www.aaai.org/ojs/index.php/AAAI/article/view/4687	t	5
1358	Near or Far, Wide Range Zero-Shot Cross-Lingual Dependency Parsing	WU Ahmad, Z Zhang, X Ma, E Hovy, KW Chang…	Cross-lingual transfer is the major means toleverage knowledge from high-resource lan-guages to help low-resource languages. In this paper, we investigate cross-lingual trans-fer across a broad spectrum of language dis-tances. We posit that Recurrent Neural Net-works …	\N	2018	https://arxiv.org/abs/1811.00570	t	5
1541	Reuse and Adaptation for Entity Resolution through Transfer Learning	S Thirumuruganathan, SAP Parambath…	Entity resolution (ER) is one of the fundamental problems in data integration, where machine learning (ML) based classifiers often provide the state-of-the-art results. Considerable human effort goes into feature engineering and training data creation. In this paper, we …	\N	2018	https://arxiv.org/abs/1809.11084	t	5
2528	Cooperative learning of disjoint syntax and semantics	S Havrylov, G Kruszewski, A Joulin	There has been considerable attention devoted to models that learn to jointly infer an expression's syntactic structure and its semantics. Yet,\\citet {NangiaB18} has recently shown that the current best systems fail to learn the correct parsing strategy on mathematical …	\N	2019	https://arxiv.org/abs/1902.09393	t	5
1416	Learning and evaluating general linguistic intelligence	D Yogatama, CM d'Autume, J Connor, T Kocisky…	We define general linguistic intelligence as the ability to reuse previously acquired knowledge about a language's lexicon, syntax, semantics, and pragmatic conventions to adapt to new tasks quickly. Using this definition, we analyze state-of-the-art natural …	\N	2019	https://arxiv.org/abs/1901.11373	t	5
2530	Multi-task Learning with Sample Re-weighting for Machine Reading Comprehension	Y Xu, X Liu, Y Shen, J Liu…	We propose a multi-task learning framework to jointly train a Machine Reading Comprehension (MRC) model on multiple datasets across different domains. Key to the proposed method is to learn robust and general contextual representations with the help of …	\N	2018	https://pdfs.semanticscholar.org/8527/e946ad292088db8b8e6084384a82299633fe.pdf	t	4
2531	Towards scalable multi-domain conversational agents: The schema-guided dialogue dataset	A Rastogi, X Zang, S Sunkara, R Gupta…	Virtual assistants such as Google Assistant, Alexa and Siri provide a conversational interface to a large number of services and APIs spanning multiple domains. Such systems need to support an ever-increasing number of services with possibly overlapping …	\N	2019	https://arxiv.org/abs/1909.05855	t	4
2532	ML-Net: multi-label classification of biomedical texts with deep neural networks	J Du, Q Chen, Y Peng, Y Xiang…	Objective In multi-label text classification, each textual document is assigned 1 or more labels. As an important task that has broad applications in biomedicine, a number of different computational methods have been proposed. Many of these methods, however, have only …	\N	2019	https://academic.oup.com/jamia/article-abstract/26/11/1279/5522430	t	4
1512	Deep Cascade Multi-task Learning for Slot Filling in Online Shopping Assistant	Y Gong, X Luo, Y Zhu, W Ou, Z Li, M Zhu…	Slot filling is a critical task in natural language understanding (NLU) for dialog systems. State-of-the-art approaches treat it as a sequence labeling problem and adopt such models as BiLSTM-CRF. While these models work relatively well on standard benchmark datasets …	\N	2019	https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4611	t	4
2549	Towards debiasing fact verification models	T Schuster, DJ Shah, YJS Yeo, D Filizzola…	Fact verification requires validating a claim in the context of evidence. We show, however, that in the popular FEVER dataset this might not necessarily be the case. Claim-only classifiers perform competitively with top evidence-aware models. In this paper, we …	\N	2019	https://arxiv.org/abs/1908.05267	t	4
1538	In-domain Context-aware Token Embeddings Improve Biomedical Named Entity Recognition	G Sheikhshab, I Birol, A Sarkar	Rapidly expanding volume of publications in the biomedical domain makes it increasingly difficult for a timely evaluation of the latest literature. That, along with a push for automated evaluation of clinical reports, present opportunities for effective natural language processing …	\N	2018	https://www.aclweb.org/anthology/W18-5618.pdf	t	4
1381	GAIA-A Multi-media Multi-lingual Knowledge Extraction and Hypothesis Generation System	T Zhang, A Subburathinam, G Shi, L Huang…	An analyst or a planner seeking a rich, deep understanding of an emergent situation today is faced with a paradox-multimodal, multilingual realtime information about most emergent situations is freely available but the sheer volume and diversity of such information makes …	\N	2018	http://nlp.cs.rpi.edu/paper/gaia2018.pdf	t	4
1338	Gradient-based inference for networks with output constraints	JY Lee, SV Mehta, M Wick, JB Tristan…	Practitioners apply neural networks to increasingly complex problems in natural language processing, such as syntactic parsing and semantic role labeling that have rich output structures. Many such structured-prediction problems require deterministic constraints on the …	\N	2017	https://arxiv.org/abs/1707.08608	t	4
2701	A pragmatic guide to geoparsing evaluation	M Gritta, MT Pilehvar, N Collier	Empirical methods in geoparsing have thus far lacked a standard evaluation framework describing the task, metrics and data used to compare state-of-the-art systems. Evaluation is further made inconsistent, even unrepresentative of real world usage by the lack of …	\N	2019	https://link.springer.com/article/10.1007/s10579-019-09475-3	t	4
2535	Alberto: Italian bert language understanding model for nlp challenging tasks based on tweets	M Polignano, P Basile, M de Gemmis…	Recent scientific studies on natural language processing (NLP) report the outstanding effectiveness observed in the use of context-dependent and task-free language understanding models such as ELMo, GPT, and BERT. Specifically, they have proved to …	\N	2019	http://ceur-ws.org/Vol-2481/paper57.pdf	t	4
2536	Unicoder: A universal language encoder by pre-training with multiple cross-lingual tasks	H Huang, Y Liang, N Duan, M Gong, L Shou…	We present Unicoder, a universal language encoder that is insensitive to different languages. Given an arbitrary NLP task, a model can be trained with Unicoder using training data in one language and directly applied to inputs of the same task in other languages …	\N	2019	https://arxiv.org/abs/1909.00964	t	4
2537	A BERT-based universal model for both within-and cross-sentence clinical temporal relation extraction	C Lin, T Miller, D Dligach, S Bethard…	Classic methods for clinical temporal relation extraction focus on relational candidates within a sentence. On the other hand, break-through Bidirectional Encoder Representations from Transformers (BERT) are trained on large quantities of arbitrary spans of contiguous text …	\N	2019	https://www.aclweb.org/anthology/W19-1908.pdf	t	4
2538	Is bert really robust? natural language attack on text classification and entailment	D Jin, Z Jin, JT Zhou, P Szolovits	Machine learning algorithms are often vulnerable to adversarial examples that have imperceptible alterations from the original counterparts but can fool the state-of-the-art models. It is helpful to evaluate or even improve the robustness of these models by exposing …	\N	2019	https://arxiv.org/abs/1907.11932	t	4
1783	Contextualized Non-local Neural Networks for Sequence Learning	P Liu, S Chang, X Huang, J Tang…	Recently, a large number of neural mechanisms and models have been proposed for sequence learning, of which self-attention, as exemplified by the Transformer model, and graph neural networks (GNNs) have attracted much attention. In this paper, we propose an …	\N	2019	https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4650	t	4
2539	One time of interaction may not be enough: Go deep with an interaction-over-interaction network for response selection in dialogues	C Tao, W Wu, C Xu, W Hu, D Zhao, R Yan	Currently, researchers have paid great attention to retrieval-based dialogues in open-domain. In particular, people study the problem by investigating context-response matching for multi-turn response selection based on publicly recognized benchmark data sets. State …	\N	2019	https://www.aclweb.org/anthology/P19-1001.pdf	t	4
1426	Revisiting the Importance of Encoding Logic Rules in Sentiment Classification	K Krishna, P Jyothi, M Iyyer	We analyze the performance of different sentiment classification models on syntactically complex inputs like A-but-B sentences. The first contribution of this analysis addresses reproducible research: to meaningfully compare different models, their accuracies must be …	\N	2018	https://arxiv.org/abs/1808.07733	t	4
2540	Don't Blame Distributional Semantics if it can't do Entailment	M Westera, G Boleda	Distributional semantics has had enormous empirical success in Computational Linguistics and Cognitive Science in modeling various semantic phenomena, such as semantic similarity, and distributional models are widely used in state-of-the-art Natural Language …	\N	2019	https://arxiv.org/abs/1905.07356	t	4
2541	Neural-Davidsonian Semantic Proto-role Labeling	R Rudinger, A Teichert, R Culkin, S Zhang…	We present a model for semantic proto-role labeling (SPRL) using an adapted bidirectional LSTM encoding strategy that we call" Neural-Davidsonian": predicate-argument structure is represented as pairs of hidden states corresponding to predicate and argument head tokens …	\N	2018	https://arxiv.org/abs/1804.07976	t	4
1367	Approaching Nested Named Entity Recognition with Parallel LSTM-CRFs	Ł Borchmann, A Gretkowski, F Gralinski	We present the winning system of this year's PolEval nested named entity competition, as well as the justification of handling the particular problem with multiple models rather than relying on dedicated architectures. The description of working out the final solution (parallel …	\N	2018	http://poleval.pl/files/poleval2018.pdf#page=63	t	4
2542	Multi-Granularity Self-Attention for Neural Machine Translation	J Hao, X Wang, S Shi, J Zhang, Z Tu	Current state-of-the-art neural machine translation (NMT) uses a deep multi-head self-attention network with no explicit phrase information. However, prior work on statistical machine translation has shown that extending the basic translation unit from words to …	\N	2019	https://arxiv.org/abs/1909.02222	t	3
1355	Recursive Routing Networks: Learning to Compose Modules for Language Understanding	I Cases, C Rosenbaum, M Riemer, A Geiger…	We introduce Recursive Routing Networks (RRNs), which are modular, adaptable models that learn effectively in diverse environments. RRNs consist of a set of functions, typically organized into a grid, and a meta-learner decision-making component called the …	\N	2019	https://www.aclweb.org/anthology/N19-1365.pdf	t	3
2702	Effective representation for easy-first dependency parsing	Z Li, J Cai, H Zhao	Easy-first parsing relies on subtree re-ranking to build the complete parse tree. Whereas the intermediate state of parsing processing is represented by various subtrees, whose internal structural information is the key lead for later parsing action decisions, we explore a better …	\N	2019	https://link.springer.com/chapter/10.1007/978-3-030-29908-8_28	t	3
2564	SG-Net: Syntax-guided machine reading comprehension	Z Zhang, Y Wu, J Zhou, S Duan, H Zhao	For machine reading comprehension, how to effectively model the linguistic knowledge from the detail-riddled and lengthy passages and get ride of the noises is essential to improve its performance. In this work, we propose using syntax to guide the text modeling of both …	\N	2019	https://arxiv.org/abs/1908.05147	t	3
2543	Semantics-aware bert for language understanding	Z Zhang, Y Wu, H Zhao, Z Li, S Zhang, X Zhou…	The latest work on language representations carefully integrates contextualized features into language model training, which enables a series of success especially in various machine reading comprehension and natural language inference tasks. However, the existing …	\N	2019	https://arxiv.org/abs/1909.02209	t	3
2544	vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations	A Baevski, S Schneider, M Auli	We propose vq-wav2vec to learn discrete representations of audio segments through a wav2vec-style self-supervised context prediction task. The algorithm uses either a gumbel softmax or online k-means clustering to quantize the dense representations. Discretization …	\N	2019	https://arxiv.org/abs/1910.05453	t	3
683	A survey on network embedding	P Cui, X Wang, J Pei, W Zhu	Network embedding assigns nodes in a network to low-dimensional representations and effectively preserves the network structure. Recently, a significant amount of progresses have been made toward this emerging network analysis paradigm. In this survey, we focus …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8392745/	t	218
2140	Charagram: Embedding words and sentences via character n-grams	J Wieting, M Bansal, K Gimpel, K Livescu	We present Charagram embeddings, a simple approach for learning character-based compositional models to embed textual sequences. A word or sentence is represented using a character n-gram count vector, followed by a single nonlinear transformation to yield a low …	\N	2016	https://arxiv.org/abs/1607.02789	t	111
2145	Improving topic models with latent feature word representations	DQ Nguyen, R Billingsley, L Du, M Johnson	Probabilistic topic models are widely used to discover latent topics in document collections, while latent feature vector representations of words have been used to obtain high performance in many NLP tasks. In this paper, we extend two different Dirichlet multinomial …	\N	2015	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00140	t	111
2147	Cross-Sentence N-ary Relation Extraction with Graph LSTMs	N Peng, H Poon, C Quirk, K Toutanova…	Past work in relation extraction has focused on binary relations in single sentences. Recent NLP inroads in high-value domains have sparked interest in the more general setting of extracting n-ary relations that span multiple sentences. In this paper, we explore a general …	\N	2017	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00049	t	111
700	Bilingual word embeddings from non-parallel document-aligned data applied to bilingual lexicon induction	I Vulic, MF Moens	We propose a simple yet effective approach to learning bilingual word embeddings (BWEs) from non-parallel document-aligned data (based on the omnipresent skip-gram model), and its application to bilingual lexicon induction (BLI). We demonstrate the utility of the induced …	\N	2015	https://lirias.kuleuven.be/1572159?limo=0	t	110
2179	Interpretable convolutional neural networks with dual local and global attention for review rating prediction	S Seo, J Huang, H Yang, Y Liu	Recently, many e-commerce websites have encouraged their users to rate shopping items and write review texts. This review information has been very useful for understanding user preferences and item properties, as well as enhancing the capability to make personalized …	\N	2017	https://dl.acm.org/citation.cfm?id=3109890	t	110
2132	Multi-perspective context matching for machine comprehension	Z Wang, H Mi, W Hamza, R Florian	Previous machine comprehension (MC) datasets are either too small to train end-to-end deep learning models, or not difficult enough to evaluate the ability of current MC techniques. The newly released SQuAD dataset alleviates these limitations, and gives us a …	\N	2016	https://arxiv.org/abs/1612.04211	t	108
2148	Improving hypernymy detection with an integrated path-based and distributional method	V Shwartz, Y Goldberg, I Dagan	Detecting hypernymy relations is a key task in NLP, which is addressed in the literature using two complementary approaches. Distributional methods, whose supervised variants are the current best performers, and path-based methods, which received less research …	\N	2016	https://arxiv.org/abs/1603.06076	t	105
2163	Linguistically regularized lstms for sentiment classification	Q Qian, M Huang, J Lei, X Zhu	Sentiment understanding has been a long-term goal of AI in the past decades. This paper deals with sentence-level sentiment classification. Though a variety of neural network models have been proposed very recently, however, previous models either depend on …	\N	2016	https://arxiv.org/abs/1611.03949	t	105
2185	Style transfer in text: Exploration and evaluation	Z Fu, X Tan, N Peng, D Zhao, R Yan	The ability to transfer styles of texts or images, is an important measurement of the advancement of artificial intelligence (AI). However, the progress in language style transfer is lagged behind other domains, such as computer vision, mainly because of the lack of …	\N	2018	https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17015	t	105
2134	Stochastic language generation in dialogue using recurrent neural networks with convolutional sentence reranking	TH Wen, M Gasic, D Kim, N Mrksic, PH Su…	The natural language generation (NLG) component of a spoken dialogue system (SDS) usually needs a substantial amount of handcrafting or a well-labeled dataset to be trained on. These limitations add significantly to development costs and make cross-domain, multi …	\N	2015	https://arxiv.org/abs/1508.01755	t	103
2172	Bidirectional long short-term memory networks for relation classification	S Zhang, D Zheng, X Hu, M Yang	Relation classification is an important semantic processing, which has achieved great attention in recent years. The main challenge is the fact that important information can appear at any position in the sentence. Therefore, we propose bidirectional long short-term …	\N	2015	https://www.aclweb.org/anthology/Y15-1009	t	103
2160	Modeling relationships in referential expressions with compositional modular networks	R Hu, M Rohrbach, J Andreas…	People often refer to entities in an image in terms of their relationships with other entities. For example," the black cat sitting under the table" refers to both a" black cat" entity and its relationship with another" table" entity. Understanding these relationships is essential for …	\N	2017	http://openaccess.thecvf.com/content_cvpr_2017/html/Hu_Modeling_Relationships_in_CVPR_2017_paper.html	t	102
715	Strategies for training large vocabulary neural language models	W Chen, D Grangier, M Auli	Training neural network language models over large vocabularies is still computationally very costly compared to count-based models such as Kneser-Ney. At the same time, neural language models are gaining popularity for many applications such as speech recognition …	\N	2015	https://arxiv.org/abs/1512.04906	t	100
2150	Noise-contrastive estimation for answer selection with deep neural networks	J Rao, H He, J Lin	We study answer selection for question answering, in which given a question and a set of candidate answer sentences, the goal is to identify the subset that contains the answer. Unlike previous work which treats this task as a straightforward pointwise classification …	\N	2016	https://dl.acm.org/citation.cfm?id=2983872	t	99
2152	Embedding-based query language models	H Zamani, WB Croft	Word embeddings, which are low-dimensional vector representations of vocabulary terms that capture the semantic similarity between them, have recently been shown to achieve impressive performance in many natural language processing tasks. The use of word …	\N	2016	https://dl.acm.org/citation.cfm?id=2970405	t	98
728	Sparse overcomplete word vector representations	M Faruqui, Y Tsvetkov, D Yogatama, C Dyer…	Current distributed representations of words show little resemblance to theories of lexical semantics. The former are dense and uninterpretable, the latter largely based on familiar, discrete classes (eg, supersenses) and relations (eg, synonymy and hypernymy). We …	\N	2015	https://arxiv.org/abs/1506.02004	t	97
697	Network embedding as matrix factorization: Unifying deepwalk, line, pte, and node2vec	J Qiu, Y Dong, H Ma, J Li, K Wang, J Tang	Since the invention of word2vec, the skip-gram model has significantly advanced the research of network embedding, such as the recent emergence of the DeepWalk, LINE, PTE, and node2vec approaches. In this work, we show that all of the aforementioned models with …	\N	2018	https://dl.acm.org/citation.cfm?id=3159706	t	192
701	The building blocks of interpretability	C Olah, A Satyanarayan, I Johnson, S Carter…	With the growing success of neural networks, there is a corresponding need to be able to explain their decisions—including building confidence about how they will behave in the real-world, detecting model bias, and for scientific curiosity. In order to do so, we need to …	\N	2018	https://distill.pub/2018/building-blocks/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=Deep%20Learning%20Weekly	t	175
770	Neural models for information retrieval	B Mitra, N Craswell	Neural ranking models for information retrieval (IR) use shallow or deep neural networks to rank search results in response to a query. Traditional learning to rank models employ machine learning techniques over hand-crafted IR features. By contrast, neural models learn …	\N	2017	https://arxiv.org/abs/1705.01509	t	97
2173	Learning text similarity with siamese recurrent networks	P Neculoiu, M Versteegh, M Rotaru	This paper presents a deep architecture for learning a similarity metric on variablelength character sequences. The model combines a stack of character-level bidirectional LSTM's with a Siamese architecture. It learns to project variablelength strings into a fixed …	\N	2016	https://www.aclweb.org/anthology/W16-1617	t	97
2191	Localizing moments in video with natural language	L Anne Hendricks, O Wang…	We consider retrieving a specific temporal segment, or moment, from a video given a natural language text description. Methods designed to retrieve whole video clips with natural language determine what occurs in a video but not when. To address this issue, we propose …	\N	2017	http://openaccess.thecvf.com/content_iccv_2017/html/Hendricks_Localizing_Moments_in_ICCV_2017_paper.html	t	97
2168	Machine translation: Mining text for social theory	JA Evans, P Aceves	More of the social world lives within electronic text than ever before, from collective activity on the web, social media, and instant messaging to online transactions, government intelligence, and digitized libraries. This supply of text has elicited demand for natural …	\N	2016	https://www.annualreviews.org/eprint/eAupeWB6qSIJttQHjXs7/full/10.1146/annurev-soc-081715-074206	t	95
729	Evaluation of word vector representations by subspace alignment	Y Tsvetkov, M Faruqui, W Ling, G Lample…	Unsupervisedly learned word vectors have proven to provide exceptionally effective features in many NLP tasks. Most common intrinsic evaluations of vector quality measure correlation with similarity judgments. However, these often correlate poorly with how well the learned …	\N	2015	https://www.aclweb.org/anthology/D15-1243	t	94
2188	Generating sentences by editing prototypes	K Guu, TB Hashimoto, Y Oren, P Liang	We propose a new generative language model for sentences that first samples a prototype sentence from the training corpus and then edits it into a new sentence. Compared to traditional language models that generate from scratch either left-to-right or by first sampling …	\N	2018	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00030	t	94
2156	Semi-supervised vocabulary-informed learning	Y Fu, L Sigal	Despite significant progress in object categorization, in recent years, a number of important challenges remain; mainly, ability to learn from limited labeled data and ability to recognize object classes within large, potentially open, set of labels. Zero-shot learning is one way of …	\N	2016	https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Fu_Semi-Supervised_Vocabulary-Informed_Learning_CVPR_2016_paper.html	t	93
2183	A hierarchical model of reviews for aspect-based sentiment analysis	S Ruder, P Ghaffari, JG Breslin	Opinion mining from customer reviews has become pervasive in recent years. Sentences in reviews, however, are usually classified independently, even though they form part of a review's argumentative structure. Intuitively, sentences in a review build and elaborate upon …	\N	2016	https://arxiv.org/abs/1609.02745	t	92
718	Improving lstm-based video description with linguistic knowledge mined from text	S Venugopalan, LA Hendricks, R Mooney…	This paper investigates how linguistic knowledge mined from large text corpora can aid the generation of natural language descriptions of videos. Specifically, we integrate both a neural language model and distributional semantics trained on large text corpora into a …	\N	2016	https://arxiv.org/abs/1604.01729	t	91
2180	Multi-view response selection for human-computer conversation	X Zhou, D Dong, H Wu, S Zhao, D Yu, H Tian…	In this paper, we study the task of response selection for multi-turn human-computer conversation. Previous approaches take word as a unit and view context and response as sequences of words. This kind of approaches do not explicitly take each utterance as a unit …	\N	2016	https://www.aclweb.org/anthology/D16-1036	t	91
2162	Generating semantically precise scene graphs from textual descriptions for improved image retrieval	S Schuster, R Krishna, A Chang, L Fei-Fei…	Semantically complex queries which include attributes of objects and relations between objects still pose a major challenge to image retrieval systems. Recent work in computer vision has shown that a graph-based semantic representation called a scene graph is an …	\N	2015	https://www.aclweb.org/anthology/W15-2812	t	90
2159	Recursive deep learning for natural language processing and computer vision	R Socher	As the amount of unstructured text data that humanity produces overall and on the Internet grows, so does the need to intelligently process it and extract different types of knowledge from it. My research goal in this thesis is to develop learning models that can automatically …	\N	2014	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.5649&rep=rep1&type=pdf	t	89
717	Phrase-based image captioning	R Lebret, PO Pinheiro, R Collobert	Generating a novel textual description of an image is an interesting problem that connects computer vision and natural language processing. In this paper, we present a simple model that is able to generate descriptive sentences given a sample image. This model has a …	\N	2015	https://arxiv.org/abs/1502.03671	t	88
734	Learning semantic word embeddings based on ordinal knowledge constraints	Q Liu, H Jiang, S Wei, ZH Ling, Y Hu	In this paper, we propose a general framework to incorporate semantic knowledge into the popular data-driven learning process of word embeddings to improve the quality of them. Under this framework, we represent semantic knowledge as many ordinal ranking …	\N	2015	https://www.aclweb.org/anthology/P15-1145	t	88
2545	Beyond English-only Reading Comprehension: Experiments in Zero-Shot Multilingual Transfer for Bulgarian	M Hardalov, I Koychev, P Nakov	Recently, reading comprehension models achieved near-human performance on large-scale datasets such as SQuAD, CoQA, MS Macro, RACE, etc. This is largely due to the release of pre-trained contextualized representations such as BERT and ELMo, which can …	\N	2019	https://arxiv.org/abs/1908.01519	t	3
1464	Effective Subword Segmentation for Text Comprehension	Z Zhang, H Zhao, K Ling, J Li, Z Li…	Character-level representations have been broadly adopted to alleviate the problem of effectively representing rare or complex words. However, character itself is not a natural minimal linguistic unit for representation or word embedding composing due to ignoring the …	\N	2019	https://ieeexplore.ieee.org/abstract/document/8735719/	t	3
2546	On the word alignment from neural machine translation	X Li, G Li, L Liu, M Meng, S Shi	Prior researches suggest that neural machine translation (NMT) captures word alignment through its attention mechanism, however, this paper finds attention may almost fail to capture word alignment for some NMT models. This paper thereby proposes two methods to …	\N	2019	https://www.aclweb.org/anthology/P19-1124.pdf	t	3
707	Context-dependent sentiment analysis in user-generated videos	S Poria, E Cambria, D Hazarika, N Majumder…	Multimodal sentiment analysis is a developing area of research, which involves the identification of sentiments in videos. Current research considers utterances as independent entities, ie, ignores the interdependencies and relations among the utterances of a video. In …	\N	2017	https://www.aclweb.org/anthology/papers/P/P17/P17-1081/	t	147
709	Toward optimal feature selection in naive Bayes for text categorization	B Tang, S Kay, H He	Automated feature selection is important for text categorization to reduce feature size and to speed up learning process of classifiers. In this paper, we present a novel and efficient feature selection framework based on the Information Theory, which aims to rank the …	\N	2016	https://ieeexplore.ieee.org/abstract/document/7465795/	t	119
710	Hierarchical recurrent neural network for document modeling	R Lin, S Liu, M Yang, M Li, M Zhou, S Li	This paper proposes a novel hierarchical recurrent neural network language model (HRNNLM) for document modeling. After establishing a RNN to capture the coherence between sentences in a document, HRNNLM integrates it as the sentence history …	\N	2015	https://www.aclweb.org/anthology/D15-1106	t	101
708	Disan: Directional self-attention network for rnn/cnn-free language understanding	T Shen, T Zhou, G Long, J Jiang, S Pan…	Recurrent neural nets (RNN) and convolutional neural nets (CNN) are widely used on NLP tasks to capture the long-term and local dependencies, respectively. Attention mechanisms have recently attracted enormous interest due to their highly parallelizable computation …	\N	2018	https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16126	t	203
706	How to train good word embeddings for biomedical NLP	B Chiu, G Crichton, A Korhonen, S Pyysalo	The quality of word embeddings depends on the input corpora, model architectures, and hyper-parameter settings. Using the state-of-the-art neural embedding tool word2vec and both intrinsic and extrinsic evaluations, we present a comprehensive study of how the quality …	\N	2016	https://www.aclweb.org/anthology/W16-2922	t	157
716	Visual question answering: A survey of methods and datasets	Q Wu, D Teney, P Wang, C Shen, A Dick…	Visual Question Answering (VQA) is a challenging task that has received increasing attention from both the computer vision and the natural language processing communities. Given an image and a question in natural language, it requires reasoning over visual …	\N	2017	https://www.sciencedirect.com/science/article/pii/S1077314217300772	t	124
2184	Making neural qa as simple as possible but not simpler	D Weissenborn, G Wiese, L Seiffe	Recent development of large-scale question answering (QA) datasets triggered a substantial amount of research into end-to-end neural architectures for QA. Increasingly complex systems have been conceived without comparison to simpler neural baseline …	\N	2017	https://arxiv.org/abs/1703.04816	t	88
2178	Relevance-based word embedding	H Zamani, WB Croft	Learning a high-dimensional dense representation for vocabulary terms, also known as a word embedding, has recently attracted much attention in natural language processing and information retrieval tasks. The embedding vectors are typically learned based on term …	\N	2017	https://dl.acm.org/citation.cfm?id=3080831	t	87
2165	NeuroNER: an easy-to-use program for named-entity recognition based on neural networks	F Dernoncourt, JY Lee, P Szolovits	Named-entity recognition (NER) aims at identifying entities of interest in a text. Artificial neural networks (ANNs) have recently been shown to outperform existing NER systems. However, ANNs remain challenging to use for non-expert users. In this paper, we present …	\N	2017	https://arxiv.org/abs/1705.05487	t	86
2176	emoji2vec: Learning emoji representations from their description	B Eisner, T Rocktäschel, I Augenstein…	Many current natural language processing applications for social media rely on representation learning and utilize pre-trained word embeddings. There currently exist several publicly-available, pre-trained sets of word embeddings, but they contain few or no …	\N	2016	https://arxiv.org/abs/1609.08359	t	86
724	Unsupervised morphology induction using word embeddings	R Soricut, F Och	We present a language agnostic, unsupervised method for inducing morphological transformations between words. The method relies on certain regularities manifest in highdimensional vector spaces. We show that this method is capable of discovering a wide …	\N	2015	https://www.aclweb.org/anthology/N15-1186	t	85
2177	Data ex machina: introduction to big data	D Lazer, J Radford	Social life increasingly occurs in digital environments and continues to be mediated by digital systems. Big data represents the data being generated by the digitization of social life, which we break down into three domains: digital life, digital traces, and digitalized life. We …	\N	2017	https://www.annualreviews.org/doi/abs/10.1146/annurev-soc-060116-053457	t	85
2181	A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding.	X Zhang, H Wang	Two major tasks in spoken language understanding (SLU) are intent determination (ID) and slot filling (SF). Recurrent neural networks (RNNs) have been proved effective in SF, while there is no prior work using RNNs in ID. Based on the idea that the intent and semantic slots …	\N	2016	https://pdfs.semanticscholar.org/1f9e/2d6df1eaaf04aebf428d9fa9a9ffc89e373c.pdf	t	85
2182	Neural semantic encoders	T Munkhdalai, H Yu	We present a memory augmented neural network for natural language understanding: Neural Semantic Encoders. NSE is equipped with a novel memory update rule and has a variable sized encoding memory that evolves over time and maintains the understanding of …	\N	2017	https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5657452/	t	85
2547	CLaC at clpsych 2019: Fusion of neural features and predicted class probabilities for suicide risk assessment based on online posts	E Mohammadi, H Amini, L Kosseim	This paper summarizes our participation to the CLPsych 2019 shared task, under the name CLaC. The goal of the shared task was to detect and assess suicide risk based on a collection of online posts. For our participation, we used an ensemble method which utilizes …	\N	2019	https://www.aclweb.org/anthology/W19-3004.pdf	t	3
2548	Multi-step Entity-centric Information Retrieval for Multi-Hop Question Answering	A Godbole, D Kavarthapu, R Das, Z Gong…	Multi-hop question answering (QA) requires an information retrieval (IR) system that can find\\emph {multiple} supporting evidence needed to answer the question, making the retrieval process very challenging. This paper introduces an IR technique that uses …	\N	2019	https://arxiv.org/abs/1909.07598	t	3
2550	Overview of the NLPCC 2019 shared task: cross-domain dependency parsing	X Peng, Z Li, M Zhang, R Wang, Y Zhang…	This paper presents an overview of the NLPCC 2019 shared task on cross-domain dependency parsing, including (1) the data annotation process,(2) task settings,(3) methods, results, and analysis of submitted systems and our recent work (Li+ 19),(4) discussions on …	\N	2019	https://link.springer.com/chapter/10.1007/978-3-030-32236-6_69	t	3
731	Learning low-dimensional representations of medical concepts	Y Choi, CYI Chiu, D Sontag	We show how to learn low-dimensional representations (embeddings) of a wide range of concepts in medicine, including diseases (eg, ICD9 codes), medications, procedures, and laboratory tests. We expect that these embeddings will be useful across medical informatics …	\N	2016	https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5001761/	t	127
730	Cane: Context-aware network embedding for relation modeling	C Tu, H Liu, Z Liu, M Sun	Network embedding (NE) is playing a critical role in network analysis, due to its ability to represent vertices with efficient low-dimensional embedding vectors. However, existing NE models aim to learn a fixed context-free embedding for each vertex and neglect the diverse …	\N	2017	https://www.aclweb.org/anthology/papers/P/P17/P17-1158/	t	117
723	Enhancing deep learning sentiment analysis with ensemble techniques in social applications	O Araque, I Corcuera-Platas, JF Sanchez-Rada…	Deep learning techniques for Sentiment Analysis have become very popular. They provide automatic feature extraction and both richer representation capabilities and better performance than traditional feature based techniques (ie, surface methods). Traditional …	\N	2017	https://www.sciencedirect.com/science/article/pii/S0957417417300751	t	143
722	Sentiment analysis leveraging emotions and word embeddings	M Giatsoglou, MG Vozalis, K Diamantaras…	Sentiment analysis and opinion mining are valuable for extraction of useful subjective information out of text documents. These tasks have become of great importance, especially for business and marketing professionals, since online posted products and services …	\N	2017	https://www.sciencedirect.com/science/article/pii/S095741741630584X	t	124
726	Meta-prod2vec: Product embeddings using side-information for recommendation	F Vasile, E Smirnova, A Conneau	We propose Meta-Prod2vec, a novel method to compute item similarities for recommendation that leverages existing item metadata. Such scenarios are frequently encountered in applications such as content recommendation, ad targeting and web search …	\N	2016	https://dl.acm.org/citation.cfm?id=2959160	t	103
2169	Learning recurrent span representations for extractive question answering	K Lee, S Salant, T Kwiatkowski, A Parikh, D Das…	The reading comprehension task, that asks questions about a given evidence document, is a central problem in natural language understanding. Recent formulations of this task have typically focused on answer selection from a set of candidates pre-defined manually or …	\N	2016	https://arxiv.org/abs/1611.01436	t	82
2161	Iterative alternating neural attention for machine reading	A Sordoni, P Bachman, A Trischler…	We propose a novel neural attention architecture to tackle machine comprehension tasks, such as answering Cloze-style queries with respect to a document. Unlike previous models, we do not collapse the query into a single vector, instead we deploy an iterative alternating …	\N	2016	https://arxiv.org/abs/1606.02245	t	81
2167	One vector is not enough: Entity-augmented distributed semantics for discourse relations	Y Ji, J Eisenstein	Discourse relations bind smaller linguistic units into coherent texts. Automatically identifying discourse relations is difficult, because it requires understanding the semantics of the linked arguments. A more subtle challenge is that it is not enough to represent the meaning of each …	\N	2015	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00142	t	80
2186	Dependency sensitive convolutional neural networks for modeling sentences and documents	R Zhang, H Lee, D Radev	The goal of sentence and document modeling is to accurately represent the meaning of sentences and documents for various Natural Language Processing tasks. In this work, we present Dependency Sensitive Convolutional Neural Networks (DSCNN) as a general …	\N	2016	https://arxiv.org/abs/1611.02361	t	78
2187	Recursive neural networks can learn logical semantics	SR Bowman, C Potts, CD Manning	Tree-structured recursive neural networks (TreeRNNs) for sentence meaning have been successful for many applications, but it remains an open question whether the fixed-length representations that they learn can support tasks as demanding as logical deduction. We …	\N	2014	https://arxiv.org/abs/1406.1827	t	75
2189	Fast dictionary learning with a smoothed Wasserstein loss	A Rolet, M Cuturi, G Peyré	We consider in this paper the dictionary learning problem when the observations are normalized histograms of features. This problem can be tackled using non-negative matrix factorization approaches, using typically Euclidean or Kullback-Leibler fitting errors …	\N	2016	http://www.jmlr.org/proceedings/papers/v51/rolet16.pdf	t	75
2190	Estimating embedding vectors for queries	H Zamani, WB Croft	The dense vector representation of vocabulary terms, also known as word embeddings, have been shown to be highly effective in many natural language processing tasks. Word embeddings have recently begun to be studied in a number of information retrieval (IR) …	\N	2016	https://dl.acm.org/citation.cfm?id=2970403	t	75
2196	Enhancing and combining sequential and tree lstm for natural language inference	Q Chen, X Zhu, Z Ling, S Wei…	Reasoning and inference are central to human and artificial intelligence. Modeling inference in human language is notoriously challenging but is fundamental to natural language understanding and many applications. With the availability of large annotated …	\N	2016	https://www.researchgate.net/profile/Hui_Jiang6/publication/308361705_Enhancing_and_Combining_Sequential_and_Tree_LSTM_for_Natural_Language_Inference/links/583310e808aef19cb81c8c01/Enhancing-and-Combining-Sequential-and-Tree-LSTM-for-Natural-Language-Inference.pdf	t	69
741	Learning cross-modal embeddings for cooking recipes and food images	A Salvador, N Hynes, Y Aytar, J Marin…	In this paper, we introduce Recipe1M, a new large-scale, structured corpus of over 1m cooking recipes and 800k food images. As the largest publicly available collection of recipe data, Recipe1M affords the ability to train high-capacity models on aligned, multi-modal …	\N	2017	http://openaccess.thecvf.com/content_cvpr_2017/html/Salvador_Learning_Cross-Modal_Embeddings_CVPR_2017_paper.html	t	127
1546	KNU CI System at SemEval-2018 Task4: Character Identification by Solving Sequence-Labeling Problem	C Park, H Song, C Lee	Character identification is an entity-linking task that finds words referring to the same person among the nouns mentioned in a conversation and turns them into one entity. In this paper, we define a sequence-labeling problem to solve character identification, and propose an …	\N	2018	https://www.aclweb.org/anthology/S18-1107.pdf	t	3
2551	Challenges in the construction of knowledge bases for human microbiome-disease associations	VD Badal, D Wright, Y Katsis, HC Kim…	The last few years have seen tremendous growth in human microbiome research, with a particular focus on the links to both mental and physical health and disease. Medical and experimental settings provide initial sources of information about these links, but individual …	\N	2019	https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-019-0742-2	t	3
2552	Language features matter: Effective language representations for vision-language tasks	A Burns, R Tan, K Saenko, S Sclaroff…	Shouldn't language and vision features be treated equally in vision-language (VL) tasks? Many VL approaches treat the language component as an afterthought, using simple language models that are either built upon fixed word embeddings trained on text-only data …	\N	2019	http://openaccess.thecvf.com/content_ICCV_2019/html/Burns_Language_Features_Matter_Effective_Language_Representations_for_Vision-Language_Tasks_ICCV_2019_paper.html	t	3
2553	Adversarial attacks on deep learning models in natural language processing: A survey	WE Zhang, QZ Sheng, A Alhazmi…	Authors' addresses: Wei Emma Zhang, w. zhang@ mq. edu. au, Macquarie University, Sydney, Australia, NSW 2109; Quan Z. Sheng, michael. sheng@ mq. edu. au, Macquarie University, Australia; Ahoud Alhazmi, ahoud. alhazmi@ hdr. mq. edu. au, Macquarie …	\N	2019	http://web.science.mq.edu.au/~qsheng/papers/TIST-revisedversion-2019.pdf	t	3
736	Joint learning of the embedding of words and entities for named entity disambiguation	I Yamada, H Shindo, H Takeda, Y Takefuji	Named Entity Disambiguation (NED) refers to the task of resolving multiple named entity mentions in a document to their correct references in a knowledge base (KB)(eg, Wikipedia). In this paper, we propose a novel embedding method specifically designed for NED. The …	\N	2016	https://arxiv.org/abs/1601.01343	t	117
742	Specializing word embeddings for similarity or relatedness	D Kiela, F Hill, S Clark	We demonstrate the advantage of specializing semantic word embeddings for either similarity or relatedness. We compare two variants of retrofitting and a joint-learning approach, and find that all three yield specialized semantic spaces that capture human …	\N	2015	https://www.aclweb.org/anthology/D15-1242	t	96
745	Molding cnns for text: non-linear, non-consecutive convolutions	T Lei, R Barzilay, T Jaakkola	The success of deep learning often derives from well-chosen operational building blocks. In this work, we revise the temporal convolution operation in CNNs to better adapt it to text processing. Instead of concatenating word representations, we appeal to tensor algebra and …	\N	2015	https://arxiv.org/abs/1508.04112	t	90
737	Integrating and evaluating neural word embeddings in information retrieval	G Zuccon, B Koopman, P Bruza…	Recent advances in neural language models have contributed new methods for learning distributed vector representations of words (also called word embeddings). Two such methods are the continuous bag-of-words model and the skipgram model. These methods …	\N	2015	https://dl.acm.org/citation.cfm?id=2838936	t	87
2194	Natural language comprehension with the epireader	A Trischler, Z Ye, X Yuan, K Suleman	We present the EpiReader, a novel model for machine comprehension of text. Machine comprehension of unstructured, real-world text is a major research goal for natural language processing. Current tests of machine comprehension pose questions whose answers can be …	\N	2016	https://arxiv.org/abs/1606.02270	t	66
2200	Semeval-2015 task 3: Answer selection in community question answering	P Nakov, L Màrquez, W Magdy, A Moschitti…	Community Question Answering (cQA) provides new interesting research directions to the traditional Question Answering (QA) field, eg, the exploitation of the interaction between users and the structure of related posts. In this context, we organized SemEval …	\N	2019	https://arxiv.org/abs/1911.11403	t	61
2554	Syntax-aware entity representations for neural relation extraction	Z He, W Chen, Z Li, W Zhang, H Shao, M Zhang	Distantly supervised relation extraction has been widely used to find novel relational facts between entities from text, and can be easily scaled to very large corpora. Previous studies on neural relation extraction treat this task as a multi-instance learning problem, and encode …	\N	2019	https://www.sciencedirect.com/science/article/pii/S0004370218303473	t	3
2555	Neural machine reading comprehension: Methods and trends	S Liu, X Zhang, S Zhang, H Wang, W Zhang	Machine reading comprehension (MRC), which requires a machine to answer questions based on a given context, has attracted increasing attention with the incorporation of various deep-learning techniques over the past few years. Although research on MRC …	\N	2019	https://www.mdpi.com/2076-3417/9/18/3698	t	3
2556	Reliability-aware dynamic feature composition for name tagging	Y Lin, L Liu, H Ji, D Yu, J Han	Word embeddings are widely used on a variety of tasks and can substantially improve the performance. However, their quality is not consistent throughout the vocabulary due to the long-tail distribution of word frequency. Without sufficient contexts, rare word embeddings …	\N	2019	https://www.aclweb.org/anthology/P19-1016.pdf	t	3
2557	Fast and discriminative semantic embedding	R Koopman, S Wang, G Englebienne	The embedding of words and documents in compact, semantically meaningful vector spaces is a crucial part of modern information systems. Deep Learning models are powerful but their hyperparameter selection is often complex and they are expensive to train, and while pre …	\N	2019	https://www.aclweb.org/anthology/W19-0420.pdf	t	3
2703	Robust Representation Learning of Biomedical Names	MC Phan, A Sun, Y Tay	Biomedical concepts are often mentioned in medical documents under different name variations (synonyms). This mismatch between surface forms is problematic, resulting in difficulties pertaining to learning effective representations. Consequently, this has …	\N	2019	https://www.aclweb.org/anthology/P19-1317.pdf	t	3
755	Artificial intelligence and games	GN Yannakakis, J Togelius	Library of Congress Control Number: 2018932540 © Springer International Publishing AG, part \nof Springer Nature 2018 This work is subject to copyright. All rights are reserved by the \nPublisher, whether the whole or part of the material is concerned, specifically the rights of …	\N	2018	https://link.springer.com/content/pdf/10.1007/978-3-319-63519-4.pdf	t	162
757	A survey of machine learning for big code and naturalness	M Allamanis, ET Barr, P Devanbu…	Research at the intersection of machine learning, programming languages, and software engineering has recently taken important steps in proposing learnable probabilistic models of source code that exploit the abundance of patterns of code. In this article, we survey this …	\N	2018	https://dl.acm.org/citation.cfm?id=3212695	t	141
1405	Subword Semantic Hashing for Intent Classification on Small Datasets	K Shridhar, A Dash, A Sahu…	In this paper, we introduce the use of Semantic Hashing as embedding for the task of Intent Classification and outperform previous state-of-the-art methods on three frequently used benchmarks. Intent Classification on a small dataset is a challenging task for data-hungry …	\N	2019	https://ieeexplore.ieee.org/abstract/document/8852420/	t	3
759	Metadata embeddings for user and item cold-start recommendations	M Kula	I present a hybrid matrix factorisation model representing users and items as linear combinations of their content features' latent factors. The model outperforms both collaborative and content-based models in cold-start or sparse interaction data scenarios …	\N	2015	https://arxiv.org/abs/1507.08439	t	80
1688	Transformer to CNN: Label-scarce distillation for efficient text classification	YK Chia, S Witteveen, M Andrews	Significant advances have been made in Natural Language Processing (NLP) modelling since the beginning of 2018. The new approaches allow for accurate results, even when there is little labelled data, because these NLP models can benefit from training on both task …	\N	2019	https://arxiv.org/abs/1909.03508	t	3
1292	Neural network acceptability judgments	A Warstadt, A Singh, SR Bowman	In this work, we explore the ability of artificial neural networks to judge the grammatical acceptability of a sentence. Machine learning research of this kind is well placed to answer important open questions about the role of prior linguistic bias in language acquisition by …	\N	2019	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00290	t	3
1324	Investigating the working of text classifiers	DS Sachan, M Zaheer, R Salakhutdinov	Text classification is one of the most widely studied tasks in natural language processing. Motivated by the principle of compositionality, large multilayer neural network models have been employed for this task in an attempt to effectively utilize the constituent expressions …	\N	2018	https://arxiv.org/abs/1801.06261	t	3
2558	Merge and Label: A novel neural network architecture for nested NER	J Fisher, A Vlachos	Named entity recognition (NER) is one of the best studied tasks in natural language processing. However, most approaches are not capable of handling nested structures which are common in many applications. In this paper we introduce a novel neural network …	\N	2019	https://arxiv.org/abs/1907.00464	t	3
2559	A neural virtual anchor synthesizer based on seq2seq and gan models	Z Wang, Z Liu, Z Chen, H Hu, S Lian	This paper presents a novel framework to generate realistic face video of an anchor, who is reading certain news. This task is also known as Virtual Anchor. Given some paragraphs of words, we first utilize a pretrained Word2Vec model to embed each word into a vector; then …	\N	2019	https://arxiv.org/abs/1908.07262	t	3
2704	Deep neural models for medical concept normalization in user-generated texts	Z Miftahutdinov, E Tutubalina	In this work, we consider the medical concept normalization problem, ie, the problem of mapping a health-related entity mention in a free-form text to a concept in a controlled vocabulary, usually to the standard thesaurus in the Unified Medical Language System …	\N	2019	https://arxiv.org/abs/1907.07972	t	3
2560	TheEarthIsFlat's submission to CLEF'19 CheckThat! challenge	L Favano, M Carman, P Lanzi	This report details our investigations in applying state-ofthe-art pre-trained Deep Learning models to the problems of Automated Claim Detection and Fact Checking, as part of the CLEF'19 Lab: Check-That!: Automatic Identification and Verification of Claims. The report …	\N	2019	http://ceur-ws.org/Vol-2380/paper_119.pdf	t	3
1419	Probing Biomedical Embeddings from Language Models	Q Jin, B Dhingra, WW Cohen, X Lu	Contextualized word embeddings derived from pre-trained language models (LMs) show significant improvements on downstream NLP tasks. Pre-training on domain-specific corpora, such as biomedical articles, further improves their performance. In this paper, we …	\N	2019	https://arxiv.org/abs/1904.02181	t	3
1361	Transfer learning in sentiment classification with deep neural networks	A Pagliarani, G Moro, R Pasolini…	Cross-domain sentiment classifiers aim to predict the polarity (ie sentiment orientation) of target text documents, by reusing a knowledge model learnt from a different source domain. Distinct domains are typically heterogeneous in language, so that transfer learning …	\N	2017	https://link.springer.com/chapter/10.1007/978-3-030-15640-4_1	t	3
2561	Multi-Sense embeddings through a word sense disambiguation process	T Ruas, W Grosky, A Aizawa	Natural Language Understanding has seen an increasing number of publications in the last few years, especially after robust word embeddings models became prominent, when they proved themselves able to capture and represent semantic relationships from …	\N	2019	https://www.sciencedirect.com/science/article/pii/S0957417419304269	t	3
2562	A realistic face-to-face conversation system based on deep neural networks	Z Chen, Z Liu, H Hu, J Bai, S Lian…	To improve the experiences of face-to-face conversation with avatar, this paper presents a novel conversation system. It is composed of two sequence-to-sequence models respectively for listening and speaking and a Generative Adversarial Network (GAN) based …	\N	2019	http://openaccess.thecvf.com/content_ICCVW_2019/html/ACVR/Chen_A_Realistic_Face-to-Face_Conversation_System_Based_on_Deep_Neural_Networks_ICCVW_2019_paper.html	t	2
2563	Deep Dialog Act Recognition using Multiple Token, Segment, and Context Information Representations	E Ribeiro, R Ribeiro, DM de Matos	Automatic dialog act recognition is a task that has been widely explored over the years. In recent works, most approaches to the task explored different deep neural network architectures to combine the representations of the words in a segment and generate a …	\N	2019	https://www.jair.org/index.php/jair/article/view/11594	t	2
1574	A Comparison of Context-sensitive Models for Lexical Substitution	AG Soler, A Cocos, M Apidianaki…	Word embedding representations provide good estimates of word meaning and give state-of-the art performance in semantic tasks. Embedding approaches differ as to whether and how they account for the context surrounding a word. We present a comparison of different word …	\N	2019	https://www.aclweb.org/anthology/W19-0423.pdf	t	2
2565	Extreme language model compression with optimal subwords and shared projections	S Zhao, R Gupta, Y Song, D Zhou	Pre-trained deep neural network language models such as ELMo, GPT, BERT and XLNet have recently achieved state-of-the-art performance on a variety of language understanding tasks. However, their size makes them impractical for a number of scenarios, especially on …	\N	2019	https://arxiv.org/abs/1909.11687	t	2
2566	End-to-end deep reinforcement learning based coreference resolution	H Fei, X Li, D Li, P Li	Recent neural network models have significantly advanced the task of coreference resolution. However, current neural coreference models are usually trained with heuristic loss functions that are computed over a sequence of local decisions. In this paper, we …	\N	2019	https://www.aclweb.org/anthology/P19-1064.pdf	t	2
2567	DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation	Y Zhang, S Sun, M Galley, YC Chen, C Brockett…	We present a large, tunable neural conversational response generation model, DialoGPT (dialogue generative pre-trained transformer). Trained on 147M conversation-like exchanges extracted from Reddit comment chains over a period spanning from 2005 …	\N	2019	https://arxiv.org/abs/1911.00536	t	2
768	A latent variable model approach to pmi-based word embeddings	S Arora, Y Li, Y Liang, T Ma, A Risteski	Semantic word embeddings represent the meaning of a word via a vector, and are created by diverse methods. Many use nonlinear operations on co-occurrence statistics, and have hand-tuned hyperparameters and reweighting methods. This paper proposes a new …	\N	2016	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00106	t	111
777	Simverb-3500: A large-scale evaluation set of verb similarity	D Gerz, I Vulić, F Hill, R Reichart…	Verbs play a critical role in the meaning of sentences, but these ubiquitous words have received little attention in recent distributional semantics research. We introduce SimVerb-3500, an evaluation resource that provides human ratings for the similarity of 3,500 verb …	\N	2016	https://arxiv.org/abs/1608.00869	t	109
782	A dual embedding space model for document ranking	B Mitra, E Nalisnick, N Craswell, R Caruana	A fundamental goal of search engines is to identify, given a query, documents that have relevant text. This is intrinsically difficult because the query and the document may use different vocabulary, or the document may contain query words without being relevant. We …	\N	2016	https://arxiv.org/abs/1602.01137	t	84
2568	ETNLP: A Visual-Aided Systematic Approach to Select Pre-Trained Embeddings for a Downstream Task	XS Vu, T Vu, SN Tran, L Jiang	Given many recent advanced embedding models, selecting pre-trained word embedding (aka, word representation) models best fit for a specific downstream task is non-trivial. In this paper, we propose a systematic approach, called ETNLP, for extracting, evaluating, and …	\N	2019	https://acl-bg.org/proceedings/2019/RANLP%202019/pdf/RANLP147.pdf	t	2
2569	Team yeon-zi at semeval-2019 task 4: Hyperpartisan news detection by de-noising weakly-labeled data	N Lee, Z Liu, P Fung	This paper describes our system that has been submitted to SemEval-2019 Task 4: Hyperpartisan News Detection. We focus on removing the noise inherent in the hyperpartisanship dataset from both data-level and model-level by leveraging semi …	\N	2019	https://www.aclweb.org/anthology/S19-2184.pdf	t	2
2570	Adversarial domain adaptation for machine reading comprehension	H Wang, Z Gan, X Liu, J Liu, J Gao, H Wang	In this paper, we focus on unsupervised domain adaptation for Machine Reading Comprehension (MRC), where the source domain has a large amount of labeled data, while only unlabeled passages are available in the target domain. To this end, we propose an …	\N	2019	https://arxiv.org/abs/1908.09209	t	2
1451	Multi-Granular Sequence Encoding via Dilated Compositional Units for Reading Comprehension	Y Tay, AT Luu, SC Hui	Sequence encoders are crucial components in many neural architectures for learning to read and comprehend. This paper presents a new compositional encoder for reading comprehension (RC). Our proposed encoder is not only aimed at being fast but also …	\N	2018	https://www.aclweb.org/anthology/D18-1238.pdf	t	2
2571	Predicting the Argumenthood of English Prepositional Phrases	N Kim, K Rawlins, B Van Durme…	Distinguishing between arguments and adjuncts of a verb is a longstanding, nontrivial problem. In natural language processing, argumenthood information is important in tasks such as semantic role labeling (SRL) and prepositional phrase (PP) attachment …	\N	2019	https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4626	t	2
2572	Ordinal triplet loss: Investigating sleepiness detection from speech	P Wu, SK Rallabandi, AW Black…	In this paper we present our submission to the INTERSPEECH 2019 ComParE Sleepiness challenge. By nature, the given speech dataset is an archetype of one with relatively limited samples, a complex underlying data distribution, and subjective ordinal labels. We propose …	\N	2019	https://pdfs.semanticscholar.org/17b6/74d628358864ae2548eaf41ff1c9cd384d59.pdf	t	2
1663	Bayesian Learning for Neural Dependency Parsing	E Shareghi, Y Li, Y Zhu, R Reichart…	While neural dependency parsers provide stateof-the-art accuracy for several languages, they still rely on large amounts of costly labeled training data. We demonstrate that in the small data regime, where uncertainty around parameter estimation and model prediction …	\N	2019	https://www.aclweb.org/anthology/N19-1354.pdf	t	2
2573	MultiFiT: Efficient Multi-lingual Language Model Fine-tuning	J Eisenschlos, S Ruder, P Czapla, M Kardas…	Pretrained language models are promising particularly for low-resource languages as they only require unlabelled data. However, training existing models requires huge amounts of compute, while pretrained cross-lingual models often underperform on low-resource …	\N	2019	https://arxiv.org/abs/1909.04761	t	2
2705	Pretrained transformers for simple question answering over knowledge graphs	D Lukovnikov, A Fischer, J Lehmann	Answering simple questions over knowledge graphs is a well-studied problem in question answering. Previous approaches for this task built on recurrent and convolutional neural network based architectures that use pretrained word embeddings. It was recently shown …	\N	2019	https://link.springer.com/chapter/10.1007/978-3-030-30793-6_27	t	2
2706	A representation learning framework for multi-source transfer parsing	J Guo, W Che, D Yarowsky, H Wang, T Liu	Cross-lingual model transfer has been a promising approach for inducing dependency parsers for low-resource languages where annotated treebanks are not available. The major obstacles for the model transfer approach are two-fold: 1. Lexical features are not directly transferable across languages; 2. Target language-specific syntactic structures are difficult to be recovered. To address these two challenges, we present a novel representation learning framework for multi-source transfer parsing. Our framework allows multi-source transfer …	\N	2016	https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12236	t	43
786	Deep learning for sentiment analysis: A survey	L Zhang, S Wang, B Liu	Deep learning has emerged as a powerful machine learning technique that learns multiple layers of representations or features of the data and produces state‐of‐the‐art prediction results. Along with the success of deep learning in many application domains, deep learning …	\N	2018	https://onlinelibrary.wiley.com/doi/abs/10.1002/widm.1253	t	216
793	Direct acoustics-to-word models for english conversational speech recognition	K Audhkhasi, B Ramabhadran, G Saon…	Recent work on end-to-end automatic speech recognition (ASR) has shown that the connectionist temporal classification (CTC) loss can be used to convert acoustics to phone or character sequences. Such systems are used with a dictionary and separately-trained …	\N	2017	https://arxiv.org/abs/1703.07754	t	88
792	Word embedding revisited: A new representation learning and explicit matrix factorization perspective	Y Li, L Xu, F Tian, L Jiang, X Zhong, E Chen	Recently significant advances have been witnessed in the area of distributed word representations based on neural networks, which are also known as word embeddings. Among the new word embedding models, skip-gram negative sampling (SGNS) in the …	\N	2015	https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/viewPaper/10863	t	80
784	Symmetric pattern based word embeddings for improved word similarity prediction	R Schwartz, R Reichart, A Rappoport	We present a novel word level vector representation based on symmetric patterns (SPs). For this aim we automatically acquire SPs (eg,“X and Y”) from a large corpus of plain text, and generate vectors where each coordinate represents the cooccurrence in SPs of the …	\N	2015	https://www.aclweb.org/anthology/K15-1026	t	79
2483	Roberta: A robustly optimized bert pretraining approach	Y Liu, M Ott, N Goyal, J Du, M Joshi, D Chen…	Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, hyperparameter choices have significant impact on the final results.\n\nWe present a replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size. We find that **BERT was significantly undertrained, and can match or exceed the performance of every model published after it**.\n\nOur best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the importance of previously overlooked design choices, and **raise questions about the source of recently reported improvements**. We release our models and code. \n\n\nComment\n----------\nNo use cases.	\N	2019	https://arxiv.org/abs/1907.11692	f	60
803	Gaining insights from social media language: Methodologies and challenges.	ML Kern, G Park, JC Eichstaedt, HA Schwartz…	Language data available through social media provide opportunities to study people at an unprecedented scale. However, little guidance is available to psychologists who want to enter this area of research. Drawing on tools and techniques developed in …	\N	2016	https://psycnet.apa.org/record/2016-38181-001	t	84
824	Exponential family embeddings	M Rudolph, F Ruiz, S Mandt, D Blei	Word embeddings are a powerful approach to capturing semantic similarity among terms in a vocabulary. In this paper, we develop exponential family embeddings, which extends the idea of word embeddings to other types of high-dimensional data. As examples, we studied …	\N	2016	http://papers.nips.cc/paper/6571-exponential-family-embeddings	t	75
818	Part-of-speech tagging with bidirectional long short-term memory recurrent neural network	P Wang, Y Qian, FK Soong, L He, H Zhao	Bidirectional Long Short-Term Memory Recurrent Neural Network (BLSTM-RNN) has been shown to be very effective for tagging sequential data, eg speech utterances or handwritten documents. While word embedding has been demoed as a powerful representation for …	\N	2015	https://arxiv.org/abs/1510.06168	t	67
820	A unified tagging solution: Bidirectional lstm recurrent neural network with word embedding	P Wang, Y Qian, FK Soong, L He, H Zhao	Bidirectional Long Short-Term Memory Recurrent Neural Network (BLSTM-RNN) has been shown to be very effective for modeling and predicting sequential data, eg speech utterances or handwritten documents. In this study, we propose to use BLSTM-RNN for a …	\N	2015	https://arxiv.org/abs/1511.00215	t	65
837	The role of context types and dimensionality in learning word embeddings	O Melamud, D McClosky, S Patwardhan…	We provide the first extensive evaluation of how using different types of context to learn skip-gram word embeddings affects performance on a wide range of intrinsic and extrinsic NLP tasks. Our results suggest that while intrinsic tasks tend to exhibit a clear preference to …	\N	2016	https://arxiv.org/abs/1601.00893	t	77
834	Take and took, gaggle and goose, book and read: Evaluating the utility of vector differences for lexical relation learning	E Vylomova, L Rimell, T Cohn, T Baldwin	Recent work on word embeddings has shown that simple vector subtraction over pre-trained embeddings is surprisingly effective at capturing different lexical relations, despite lacking explicit supervision. Prior work has evaluated this intriguing result using a word analogy …	\N	2015	https://arxiv.org/abs/1509.01692	t	68
476	Distributed representations of words and phrases and their compositionality	T Mikolov, I Sutskever, K Chen, GS Corrado…	The recently introduced continuous Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships.\nIn this paper we present several improvements that make the Skip-gram model more expressive and enable it to learn higher quality vectors more rapidly.\n\nWe show that by subsampling frequent words we obtain significant speedup, and also learn higher quality representations as measured by our tasks. We also introduce Negative Sampling, a simplified variant of Noise Contrastive Estimation (NCE) that learns more accurate vectors for frequent words compared to the hierarchical softmax.\n\nAn inherent limitation of word representations is their indifference to word order and their inability to represent idiomatic phrases. For example, the meanings of Canada'' and "Air'' cannot be easily combined to obtain "Air Canada''. Motivated by this example, we present a simple and efficient method for finding phrases, and show that their vector representations can be accurately learned by the Skip-gram model.\n\nComment\n----------\n\nThis paper does not itself describe many use cases of the model. The quality of the model is assessed with a self-prepared set of semantic and syntactic questions - which seems dangerous w.r.t. overfitting.\nHowever the usefulness of the model follows from the number of citations this has received.\n\nTo find the use cases of Word2Vec we must crawl the citing publications.	\N	2013	http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases	t	17270
859	Generating topical poetry	M Ghazvininejad, X Shi, Y Choi, K Knight	We describe Hafez, a program that generates any number of distinct poems on a usersupplied topic. Poems obey rhythmic and rhyme constraints. We describe the poetrygeneration algorithm, give experimental data concerning its parameters, and show its …	\N	2016	https://www.aclweb.org/anthology/D16-1126.pdf	t	70
877	Deep learning for extreme multi-label text classification	J Liu, WC Chang, Y Wu, Y Yang	Extreme multi-label text classification (XMTC) refers to the problem of assigning to each document its most relevant subset of class labels from an extremely large label collection, where the number of labels could reach hundreds of thousands or millions. The huge label …	\N	2017	https://dl.acm.org/citation.cfm?id=3080834	t	125
880	Learning to segment every thing	R Hu, P Dollár, K He, T Darrell…	Most methods for object instance segmentation require all training examples to be labeled with segmentation masks. This requirement makes it expensive to annotate new categories and has restricted instance segmentation models to~ 100 well-annotated classes. The goal …	\N	2018	http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Learning_to_Segment_CVPR_2018_paper.html	t	101
518	Retrofitting word vectors to semantic lexicons	M Faruqui, J Dodge, SK Jauhar, C Dyer, E Hovy…	Vector space word representations are learned from distributional information of words in large corpora.\nAlthough such statistics are semantically informative, they disregard the valuable information that is contained in semantic lexicons such as WordNet, FrameNet, and the Paraphrase Database.\n\nThis paper proposes a method for **refining vector space representations using relational information from semantic lexicons by encouraging linked words to have similar vector representations**, and it makes no assumptions about how the input vectors were constructed.\n\nEvaluated on a battery of standard lexical semantic evaluation tasks in several languages, we obtain substantial improvements starting with a variety of word vector models. Our refinement method outperforms prior techniques for incorporating semantic lexicons into the word vector training algorithms. 	\N	2014	https://arxiv.org/abs/1411.4166	t	545
930	Word embeddings quantify 100 years of gender and ethnic stereotypes	N Garg, L Schiebinger, D Jurafsky…	Word embeddings are a powerful machine-learning framework that represents each English word by a vector. The geometric relationship between these vectors captures meaningful semantic relationships between the corresponding words. In this paper, we develop a …	\N	2018	https://www.pnas.org/content/115/16/E3635.short	t	125
497	Character-aware neural language models	Y Kim, Y Jernite, D Sontag, AM Rush	We describe a simple neural language model that relies only on character-level inputs. Predictions are still made at the word-level. Our model employs a convolutional neural network (CNN) and a highway network over characters, whose output is given to a long short-term memory (LSTM) recurrent neural network language model (RNN-LM). On the English Penn Treebank the model is on par with the existing state-of-the-art despite having 60% fewer parameters. On languages with rich morphology (Arabic, Czech, French, German, Spanish, Russian), the model outperforms word-level/morpheme-level LSTM baselines, again with fewer parameters. The results suggest that on many languages, character inputs are sufficient for language modeling. Analysis of word representations obtained from the character composition part of the model reveals that the model is able to encode, from characters only, both semantic and orthographic information. \n\nComment\n----------\n\nDo not compute word embeddings.	\N	2016	https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12489	f	1034
1794	Deep contextualized word representations	ME Peters, M Neumann, M Iyyer, M Gardner…	We introduce a new type of deep contextualized word representation that models both\n\n 1. complex characteristics of word use (e.g., syntax and semantics), and\n 2. how these uses vary across linguistic contexts (i.e., to model polysemy).\n\nOur word vectors are learned **functions of the internal states of a deep bidirectional language model (biLM)**, which is pretrained on a large text corpus.\n\nWe show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including\n\n - question answering\n - textual entailment and\n - sentiment analysis.\n\nWe also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.	\N	2018	https://arxiv.org/abs/1802.05365	t	2104
508	Convolutional neural network architectures for matching natural language sentences	B Hu, Z Lu, H Li, Q Chen	Semantic matching is of central importance to many natural language tasks\\cite {bordes2014semantic, RetrievalQA}. A successful matching algorithm needs to adequately model the internal structures of language objects and the interaction between them. As a …\n\nComment\n----------\n\nNot really about embeddings.	\N	2014	http://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matc	f	761
1342	Neo: A learned query optimizer	R Marcus, P Negi, H Mao, C Zhang, M Alizadeh…	Query optimization is one of the most challenging problems in database systems. Despite the progress made over the past decades, query optimizers remain extremely complex components that require a great deal of hand-tuning for specific workloads and datasets. \n\nMotivated by this shortcoming and inspired by recent advances in applying machine learning to data management challenges, we introduce Neo (Neural Optimizer), a novel learning-based query optimizer that relies on deep neural networks to generate query executions plans. Neo bootstraps its query optimization model from existing optimizers and continues to learn from incoming queries, building upon its successes and learning from its failures. Furthermore, Neo naturally adapts to underlying data patterns and is robust to estimation errors.\n\nExperimental results demonstrate that Neo, even when bootstrapped from a simple optimizer like PostgreSQL, can learn a model that offers similar performance to state-of-the-art commercial optimizers, and in some cases even surpass them.	\N	2019	https://arxiv.org/abs/1904.03711	t	19
2053	Multimodal compact bilinear pooling for visual question answering and visual grounding	A Fukui, DH Park, D Yang, A Rohrbach…	Modeling textual or visual information with vector representations trained from large language or visual datasets has been successfully explored in recent years. However, tasks such as visual question answering require combining these vector representations with …	\N	2016	https://arxiv.org/abs/1606.01847	t	589
2074	A thorough examination of the cnn/daily mail reading comprehension task	D Chen, J Bolton, CD Manning	Enabling a computer to understand a document so that it can answer comprehension questions is a central, yet unsolved goal of NLP. A key factor impeding its solution by machine learned systems is the limited availability of human-annotated data. Hermann et …	\N	2016	https://arxiv.org/abs/1606.02858	t	307
1269	Neural approaches to conversational AI	J Gao, M Galley, L Li	The present paper surveys neural approaches to conversational AI that have been developed in the last few years. We group conversational systems into three categories:\n\n 1. question answering agents,\n 2. task-oriented dialogue agents, and\n 3. chatbots.\n\nFor each category, we present a review of state-of-the-art neural approaches, draw the connection between them and traditional approaches, and discuss the progress that has been made and challenges still being faced, using specific systems and models as case studies.\n\nComment\n----------\n\nNo use cases of embeddings.	\N	2019	http://www.nowpublishers.com/article/Details/INR-074	f	95
494	Neural architectures for named entity recognition	G Lample, M Ballesteros, S Subramanian…	State-of-the-art named entity recognition systems rely heavily on hand-crafted features and domain-specific knowledge in order to learn effectively from the small, supervised training corpora that are available.\nIn this paper, we introduce two new neural architectures - one based on bidirectional LSTMs and conditional random fields, and the other that constructs and labels segments using a transition-based approach inspired by shift-reduce parsers.\n\nOur models rely on two sources of information about words: character-based word representations learned from the supervised corpus and unsupervised word representations learned from unannotated corpora. Our models obtain state-of-the-art performance in NER in four languages without resorting to any language-specific knowledge or resources such as gazetteers. 	\N	2016	https://arxiv.org/abs/1603.01360	t	1474
496	Bag of tricks for efficient text classification	A Joulin, E Grave, P Bojanowski, T Mikolov	This paper explores a simple and efficient baseline for text classification. Our experiments show that our fast text classifier **fastText is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster** for training and evaluation.\nWe can train fastText on more than one billion words in less than ten minutes using a standard multicore~CPU, and classify half a million sentences among~312K classes in less than a minute.\n\nComment\n----------\n\nDiscarded: no new use cases	\N	2016	https://arxiv.org/abs/1607.01759	f	1446
2052	Named entity recognition with bidirectional LSTM-CNNs	JPC Chiu, E Nichols	Named entity recognition is a challenging task that has traditionally required large amounts of knowledge in the form of feature engineering and lexicons to achieve high performance.\n\nIn this paper, we present a novel neural network architecture that automatically detects word- and character-level features using a hybrid bidirectional LSTM and CNN architecture, eliminating the need for most feature engineering. We also propose a novel method of encoding partial lexicon matches in neural networks and compare it to existing approaches.\n\nExtensive evaluation shows that, given only tokenized text and publicly available word embeddings, our system is competitive on the CoNLL-2003 dataset and surpasses the previously reported state of the art performance on the OntoNotes 5.0 dataset by 2.13 F1 points. By using two lexicons constructed from publicly-available sources, we establish new state of the art performance with an F1 score of 91.62 on CoNLL-2003 and 86.28 on OntoNotes, surpassing systems that employ heavy feature engineering, proprietary lexicons, and rich entity linking information.	\N	2016	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00104	t	689
517	Deep learning with differential privacy	M Abadi, A Chu, I Goodfellow, HB McMahan…	Machine learning techniques based on neural networks are achieving remarkable results in a wide variety of domains. Often, the training of models requires large, representative datasets, which may be crowdsourced and contain sensitive information. The …\n\nComment\n----------\nNo use cases for WEMs.	\N	2016	https://dl.acm.org/citation.cfm?id=2978318	f	779
477	Imagenet large scale visual recognition challenge	O Russakovsky, J Deng, H Su, J Krause…	The ImageNet Large Scale Visual Recognition Challenge is a benchmark in object category classification and detection on hundreds of object categories and millions of images. The challenge has been run annually from 2010 to present, attracting participation …\n\nComment\n----------\n\nNot about word embeddings.	\N	2015	https://link.springer.com/article/10.1007/s11263-015-0816-y	f	14071
507	Visual genome: Connecting language and vision using crowdsourced dense image annotations	R Krishna, Y Zhu, O Groth, J Johnson, K Hata…	Despite progress in perceptual tasks such as image classification, computers still perform poorly on cognitive tasks such as image description and question answering. Cognition is core to tasks that involve not just recognizing, but reasoning about our visual world. However, models used to tackle the rich content in images for cognitive tasks are still being trained using the same datasets designed for perceptual tasks. To achieve success at cognitive tasks, models need to understand the interactions and relationships between objects in an image. When asked “What vehicle is the person riding?”, computers will need to identify the objects in an image as well as the relationships riding(man, carriage) and pulling(horse, carriage) to answer correctly that “the person is riding a horse-drawn carriage.” In this paper, we present the Visual Genome dataset to enable the modeling of such relationships.\n\nWe **collect dense annotations of objects, attributes, and relationships within each image to learn these models**. Specifically, our dataset contains over 108K images where each image has an average of 35 objects, 26 attributes, and 21 pairwise relationships between objects. We canonicalize the objects, attributes, relationships, and noun phrases in region descriptions and questions answer pairs to WordNet synsets. Together, these annotations represent the densest and largest dataset of image descriptions, objects, attributes, relationships, and question answer pairs.\n	\N	2017	https://link.springer.com/article/10.1007/S11263-016-0981-7	t	941
652	Normalized word embedding and orthogonal transform for bilingual word translation	C Xing, D Wang, C Liu, Y Lin	Word embedding has been found to be highly powerful to translate words from one language to another by a simple linear transform. However, we found some inconsistence among the objective functions of the embedding and the transform learning, as well as the distance measurement.\n\nThis paper proposes a solution which **normalizes the word vectors on a hypersphere and constrains the linear transform as an orthogonal transform**. The experimental results confirmed that the proposed solution can offer better performance on a word similarity task and an English-to-Spanish word translation task.	\N	2015	https://www.aclweb.org/anthology/N15-1104	t	180
2054	A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification	Y Zhang, B Wallace	Convolutional Neural Networks (CNNs) have recently achieved remarkably strong performance on the practically important task of sentence classification (kim 2014, kalchbrenner 2014, johnson 2014). However, these models require practitioners to specify …	\N	2015	https://arxiv.org/abs/1510.03820	t	529
1270	Reinforced mnemonic reader for machine reading comprehension	M Hu, Y Peng, Z Huang, X Qiu, F Wei…	In this paper, we introduce the Reinforced Mnemonic Reader for machine reading comprehension tasks, which enhances previous attentive readers in two aspects.\n\nFirst, a reattention mechanism is proposed to refine current attentions by directly accessing to past attentions that are temporally memorized in a multi-round alignment architecture, so as to avoid the problems of attention redundancy and attention deficiency.\n\nSecond, a new optimization approach, called dynamic-critical reinforcement learning, is introduced to extend the standard supervised method. It always encourages to predict a more acceptable answer so as to address the convergence suppression problem occurred in traditional reinforcement learning algorithms. Extensive experiments on the Stanford Question Answering Dataset (SQuAD) show that our model achieves state-of-the-art results. Meanwhile, our model outperforms previous systems by over 6% in terms of both Exact Match and F1 metrics on two adversarial SQuAD datasets. 	\N	2017	https://arxiv.org/abs/1705.02798	t	68
484	Intriguing properties of neural networks	C Szegedy, W Zaremba, I Sutskever, J Bruna…	Deep neural networks are highly expressive models that have recently achieved state of the art performance on speech and visual recognition tasks. While their expressiveness is the reason they succeed, it also causes them to learn uninterpretable solutions that could have counter-intuitive properties. In this paper we report two such properties.\n\nFirst, we find that there is no distinction between individual high level units and random linear combinations of high level units, according to various methods of unit analysis. It suggests that it is the space, rather than the individual units, that contains of the semantic information in the high layers of neural networks.\n\nSecond, we find that **deep neural networks learn input-output mappings that are fairly discontinuous to a significant extend**.\n**We can cause the network to misclassify an image by applying a certain imperceptible perturbation**, which is found by maximizing the network's prediction error. In addition, the specific nature of these perturbations is not a random artifact of learning: the **same perturbation can cause a different network, that was trained on a different subset of the dataset, to misclassify the same input.**	\N	2013	https://arxiv.org/abs/1312.6199	f	3596
485	Deepwalk: Online learning of social representations	B Perozzi, R Al-Rfou, S Skiena	We present DeepWalk, a novel approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements …\n\nComment\n----------\n\nDiscarded: not a use case of word embeddings, but of embeddings in general	\N	2014	https://dl.acm.org/citation.cfm?id=2623732	f	2625
487	Conditional generative adversarial nets	M Mirza, S Osindero	Generative Adversarial Nets [8] were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the …	\N	2014	https://arxiv.org/abs/1411.1784	f	2581
486	Deep learning: methods and applications	L Deng, D Yu	This monograph provides an overview of general deep learning methodology and its applications to a variety of signal and information processing tasks. The application areas are chosen with the following three criteria in mind:(1) expertise or knowledge of the …	\N	2014	http://www.nowpublishers.com/article/Details/SIG-039	f	2131
488	node2vec: Scalable feature learning for networks	A Grover, J Leskovec	Prediction tasks over nodes and edges in networks require careful effort in engineering features used by learning algorithms. Recent research in the broader field of representation learning has led to significant progress in automating prediction by learning the features …	\N	2016	https://dl.acm.org/citation.cfm?id=2939754	f	2442
2056	Supervised learning of universal sentence representations from natural language inference data	A Conneau, D Kiela, H Schwenk, L Barrault…	Many modern NLP systems rely on word embeddings, previously trained in an unsupervised manner on large corpora, as base features. Efforts to obtain embeddings for larger chunks of text, such as sentences, have however not been so successful. Several attempts at learning …	\N	2017	https://arxiv.org/abs/1705.02364	t	622
1279	Language models are unsupervised multitask learners	A Radford, J Wu, R Child, D Luan, D Amodei…	Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets.\n\nWe demonstrate that **language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText**. When conditioned on a document plus questions, the answers generated by the language model reach 55 F1 on the CoQA dataset - matching or exceeding the performance of 3 out of 4 baseline systems without using the 127,000+ training examples.\n\nThe capacity of the language model is essential to the success of zero-shot task transfer and increasing it improves performance in a log-linear fashion across tasks. Our largest model, GPT-2, is a 1.5B parameter Transformer that achieves state of the art results on 7 out of 8 tested language modeling datasets in a zero-shot setting but still underfits WebText. Samples from the model reflect these improvements and contain coherent paragraphs of text. These findings suggest a promising path towards building language processing systems which learn to perform tasks from their naturally occurring demonstrations. \n\nComment\n----------\n\nExtremely interesting, however not abou	\N	2019	https://www.techbooky.com/wp-content/uploads/2019/02/Better-Language-Models-and-Their-Implications.pdf	f	232
492	Devise: A deep visual-semantic embedding model	A Frome, GS Corrado, J Shlens, S Bengio…	Modern visual recognition systems are often limited in their ability to scale to large numbers of object categories. This limitation is in part due to the increasing difficulty of acquiring sufficient training data in the form of labeled images as the number of object categories grows. One remedy is to leverage data from other sources -- such as text data -- both to train visual models and to constrain their predictions.\n\nIn this paper we present a new deep visual-semantic embedding model **trained to identify visual objects using both labeled image data as well as semantic information gleaned from unannotated text**.\n\nWe demonstrate that this model matches state-of-the-art performance on the 1000-class ImageNet object recognition challenge while making more semantically reasonable errors, and also show that the semantic information can be exploited to make predictions about tens of thousands of image labels not observed during training. Semantic knowledge improves such zero-shot predictions by up to 65%, achieving hit rates of up to 10% across thousands of novel labels never seen by the visual model.	\N	2013	http://papers.nips.cc/paper/5204-devise-a-deep-visual-sem	t	1286
2051	Inductive representation learning on large graphs	W Hamilton, Z Ying, J Leskovec	Low-dimensional embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks, from content recommendation to identifying protein functions. However, most existing approaches require that all nodes in the graph are present during …	\N	2017	http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs	t	982
2050	Image captioning with semantic attention	Q You, H Jin, Z Wang, C Fang…	Automatically generating a natural language description of an image has attracted interests recently both because of its importance in practical applications and because it connects two major artificial intelligence fields: computer vision and natural language processing. Existing …	\N	2016	http://openaccess.thecvf.com/content_cvpr_2016/html/You_Image_Captioning_With_CVPR_2016_paper.html	t	710
2079	Asymmetric transitivity preserving graph embedding	M Ou, P Cui, J Pei, Z Zhang, W Zhu	Graph embedding algorithms embed a graph into a vector space where the structure and the inherent properties of the graph are preserved. The existing graph embedding methods cannot preserve the asymmetric transitivity well, which is a critical property of directed …	\N	2016	https://dl.acm.org/citation.cfm?id=2939751	t	351
2149	Recurrent attention network on memory for aspect sentiment analysis	P Chen, Z Sun, L Bing, W Yang	We propose a novel framework based on neural networks to identify the sentiment of opinion targets in a comment/review. Our framework adopts multiple-attention mechanism to capture sentiment features separated by a long distance, so that it is more robust against …	\N	2017	https://www.aclweb.org/anthology/papers/D/D17/D17-1047/	t	171
1411	Transformer-xl: Attentive language models beyond a fixed-length context	Z Dai, Z Yang, Y Yang, WW Cohen, J Carbonell…	Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the setting of language modeling.\n\nWe propose a novel neural architecture Transformer-XL that enables learning dependency beyond a fixed length without disrupting temporal coherence. It consists of a segment-level recurrence mechanism and a novel positional encoding scheme. Our method not only enables capturing longer-term dependency, but also resolves the context fragmentation problem.\n\nAs a result, Transformer-XL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+ times faster than vanilla Transformers during evaluation. Notably, we improve the state-of-the-art results of bpc/perplexity to 0.99 on enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5 on Penn Treebank (without finetuning). When trained only on WikiText-103, Transformer-XL manages to generate reasonably coherent, novel text articles with thousands of tokens. Our code, pretrained models, and hyperparameters are available in both Tensorflow and PyTorch. \n\nComment\n----------\n\nNo use cases of corresembeddings.	\N	2019	https://arxiv.org/abs/1901.02860	f	196
2155	Interactive attention networks for aspect-level sentiment classification	D Ma, S Li, X Zhang, H Wang	Aspect-level sentiment classification aims at identifying the sentiment polarity of specific target in its context. Previous approaches have realized the importance of targets in sentiment classification and developed various methods with the goal of precisely modeling …	\N	2017	https://arxiv.org/abs/1709.00893	t	170
2153	Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs	YA Malkov, DA Yashunin	We present a new approach for the approximate K-nearest neighbor search based on navigable small world graphs with controllable hierarchy (Hierarchical NSW, HNSW). The proposed solution is fully graph-based, without any need for additional search structures …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8594636/	t	136
501	Deep convolutional neural networks for sentiment analysis of short texts	C Dos Santos, M Gatti	Sentiment analysis of short texts such as single sentences and Twitter messages is challenging because of the limited contextual information that they normally contain. Effectively solving this task requires strategies that combine the small text content with prior knowledge and use more than just bag-of-words.\n\nIn this work we propose a new deep convolutional neural network that exploits from character-to sentence-level information to perform sentiment analysis of short texts. We apply our approach for two corpora of two different domains: the Stanford Sentiment Tree-bank (SSTb), which contains sentences from movie reviews; and the Stanford Twitter Sentiment corpus (STS), which contains Twitter messages.\n\nFor the SSTb corpus, our approach achieves state-of-the-art results for single sentence sentiment prediction in both binary positive/negative classification, with 85.7% accuracy, and fine-grained classification, with 48.3% accuracy. For the STS corpus, our approach achieves a sentiment prediction accuracy of 86.4%.	\N	2014	https://www.aclweb.org/anthology/C14-1008	t	906
2197	Natural language inference over interaction space	Y Gong, H Luo, J Zhang	Natural Language Inference (NLI) task requires an agent to determine the logical relationship between a natural language premise and a natural language hypothesis. We introduce Interactive Inference Network (IIN), a novel class of neural network architectures …	\N	2017	https://arxiv.org/abs/1709.04348	t	91
2193	Graph-structured representations for visual question answering	D Teney, L Liu…	This paper proposes to improve visual question answering (VQA) with structured representations of both scene contents and questions. A key challenge in VQA is to require joint reasoning over the visual and text domains. The predominant CNN/LSTM-based …	\N	2017	http://openaccess.thecvf.com/content_cvpr_2017/html/Teney_Graph-Structured_Representations_for_CVPR_2017_paper.html	t	89
2195	R 3: Reinforced ranker-reader for open-domain question answering	S Wang, M Yu, X Guo, Z Wang, T Klinger…	In recent years researchers have achieved considerable success applying neural network methods to question answering (QA). These approaches have achieved state of the art results in simplified closed-domain settings such as the SQuAD (Rajpurkar et al. 2016) …	\N	2018	https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16712	t	88
2199	Procedural content generation via machine learning (PCGML)	A Summerville, S Snodgrass, M Guzdial…	This survey explores procedural content generation via machine learning (PCGML), defined as the generation of game content using machine learning models trained on existing content. As the importance of PCG for game development increases, researchers explore …	\N	2018	https://ieeexplore.ieee.org/abstract/document/8382283/	t	87
2192	Learning to compose words into sentences with reinforcement learning	D Yogatama, P Blunsom, C Dyer, E Grefenstette…	We use reinforcement learning to learn tree-structured neural networks for computing representations of natural language sentences. In contrast with prior work on tree-structured models in which the trees are either provided as input or predicted using supervision from …	\N	2016	https://arxiv.org/abs/1611.09100	t	75
2198	Attention-based convolutional neural network for machine comprehension	W Yin, S Ebert, H Schütze	Understanding open-domain text is one of the primary challenges in natural language processing (NLP). Machine comprehension benchmarks evaluate the system's ability to understand text based on the text content only. In this work, we investigate machine …	\N	2016	https://arxiv.org/abs/1602.04341	t	67
482	Distributed representations of sentences and documents	Q Le, T Mikolov	Many machine learning algorithms require the input to be represented as a fixed-length feature vector. When it comes to texts, one of the most common fixed-length features is bag-of-words.\nDespite their popularity, bag-of-words features have two major weaknesses: they lose the ordering of the words and they also ignore semantics of the words. For example, "powerful," "strong" and "Paris" are equally distant.\n\nIn this paper, we propose Paragraph Vector, an unsupervised algorithm that **learns fixed-length feature representations from variable-length pieces of texts**, such as sentences, paragraphs, and documents. Our algorithm represents each document by a dense vector which is trained to predict words in the document. Its construction gives our algorithm the potential to overcome the weaknesses of bag-of-words models. Empirical results show that **Paragraph Vectors outperform bag-of-words models** as well as other techniques for text representations. Finally, we achieve new **state-of-the-art results on several text classification and sentiment analysis tasks**. 	\N	2014	http://www.jmlr.org/proceedings/papers/v32/le14.pdf	t	5038
2246	Contextual string embeddings for sequence labeling	A Akbik, D Blythe, R Vollgraf	Recent advances in language modeling using recurrent neural networks have made it viable to model language as distributions over characters. By learning to predict the next character on the basis of previous characters, such models have been shown to …	\N	2018	https://www.aclweb.org/anthology/C18-1139.pdf	t	196
505	From word embeddings to document distances	M Kusner, Y Sun, N Kolkin, K Weinberger	We present the Word Mover's Distance (WMD), a novel distance function between text documents. Our work is based on recent results in word embeddings that learn semantically meaningful representations for words from local cooccurrences in sentences.\n\nThe WMD distance measures the dissimilarity between two text documents as the minimum amount of distance that the embedded words of one document need to "travel" to reach the embedded words of another document.\n\nWe show that this distance metric can be cast as an instance of the Earth Mover's Distance, a well studied transportation problem for which several highly efficient solvers have been developed. Our metric has no hyperparameters and is straight-forward to implement. Further, we demonstrate on eight real world document classification data sets, in comparison with seven state-of-the-art baselines, that the **WMD metric leads to unprecedented low k-nearest neighbor document classification error rates**.	\N	2015	http://www.jmlr.org/proceedings/papers/v37/kusnerb15.pdf	t	878
523	Zero-shot learning by convex combination of semantic embeddings	M Norouzi, T Mikolov, S Bengio, Y Singer…	Several recent publications have proposed methods for mapping images into continuous semantic embedding spaces. In some cases the embedding space is trained jointly with the image transformation. In other cases the semantic embedding space is established by an independent natural language processing task, and then the image transformation into that space is learned in a second stage.\n\nProponents of these image embedding systems have stressed their advantages over the traditional classification framing of image understanding, particularly in terms of the promise for zero-shot learning - the ability to correctly annotate images of previously unseen object categories.\n\nIn this paper, we propose a simple method for constructing an image embedding system from any existing image classifier and a semantic word embedding model, which contains the n class labels in its vocabulary. Our method **maps images into the semantic embedding space via convex combination of the class label embedding vectors**, and **requires no additional training**.\nWe show that this simple and direct method confers many of the advantages associated with more complex image embedding schemes, and indeed outperforms state of the art methods on the ImageNet zero-shot learning task. 	\N	2013	https://arxiv.org/abs/1312.5650	t	461
499	Improving distributional similarity with lessons learned from word embeddings	O Levy, Y Goldberg, I Dagan	Recent trends suggest that neural-network-inspired word embedding models outperform traditional count-based distributional models on word similarity and analogy detection tasks.\nWe reveal that much of the **performance gains of word embeddings are due to certain system design choices and hyperparameter optimizations, rather than the embedding algorithms themselves**. Furthermore, we show that these modifications can be transferred to traditional distributional models, yielding similar gains.\nIn contrast to prior reports, we observe **mostly local or insignificant performance differences between the methods, with no global advantage to any single approach over the others.**\n\nComment\n----------\n\nVery interesting! Motivates to also look into other models, as SVD\nBut no use cases.	\N	2015	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00134	f	899
483	Show and tell: A neural image caption generator	O Vinyals, A Toshev, S Bengio…	Automatically describing the content of an image is a fundamental problem in artificial intelligence that connects computer vision and natural language processing.\nIn this paper, we present a generative model based on a deep recurrent architecture that combines recent advances in computer vision and machine translation and that can be used to generate natural sentences describing an image. The model is **trained to maximize the likelihood of the target description sentence given the training image**.\n\nExperiments on several datasets show the accuracy of the model and the fluency of the language it learns solely from image descriptions. Our model is often quite accurate, which we verify both qualitatively and quantitatively. For instance, while the current state-of-the-art BLEU score (the higher the better) on the Pascal dataset is 25, our approach yields 59, to be compared to human performance around 69. We also show BLEU score improvements on Flickr30k, from 56 to 66, and on SBU, from 19 to 28. Lastly, on the newly released COCO dataset, we achieve a BLEU-4 of 27.7, which is the current state-of-the-art.\n\n\nComment\n----------\nThe first iteration of the LSTM has three inputs:\n\n 1. the image\n 2. the word embedding of the previously generated word\n 3. the state of the LSTM\n\nSubsequent iterations: only 2 and 3\n\nIn the end a complete sentence is generated (signalled by stop-word).	\N	2015	https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Vinyals_Show_and_Tell_2015_CVPR_paper.html	t	3159
2044	Deep visual-semantic alignments for generating image descriptions	A Karpathy, L Fei-Fei	We present a model that generates natural language descriptions of images and their regions. Our approach leverages datasets of images and their sentence descriptions to learn about the inter-modal correspondences between language and visual data.\n\nOur alignment model is based on a novel combination of Convolutional Neural Networks over image regions, bidirectional Recurrent Neural Networks over sentences, and a structured objective that aligns the two modalities through a **multimodal embedding**.\n\nWe then describe a Multimodal Recurrent Neural Network architecture that uses the inferred alignments to learn to generate novel descriptions of image regions. We demonstrate that our alignment model produces **state of the art results in retrieval experiments on Flickr8K, Flickr30K and MSCOCO datasets**. We then show that the generated descriptions significantly outperform retrieval baselines on both full images and on a new dataset of region-level annotations.\n	\N	2015	https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.html	t	3050
2046	An overview of gradient descent optimization algorithms	S Ruder	Gradient descent optimization algorithms, while increasingly popular, are often used as black-box optimizers, as practical explanations of their strengths and weaknesses are hard to come by. This article aims to provide the reader with intuitions with regard to the …	\N	2016	https://arxiv.org/abs/1609.04747	f	1567
2045	Improved semantic representations from tree-structured long short-term memory networks	KS Tai, R Socher, CD Manning	Because of their superior ability to preserve sequence information over time, Long Short-Term Memory (LSTM) networks, a type of recurrent neural network with a more complex computational unit, have obtained strong results on a variety of sequence modeling tasks. The only underlying LSTM structure that has been explored so far is a linear chain.\n\nHowever, **natural language exhibits syntactic properties that would naturally combine words to phrases.** We introduce the Tree-LSTM, a generalization of LSTMs to tree-structured network topologies. **Tree-LSTMs outperform all existing systems and strong LSTM baselines on two tasks**: predicting the semantic relatedness of two sentences (SemEval 2014, Task 1) and sentiment classification (Stanford Sentiment Treebank). 	\N	2015	https://arxiv.org/abs/1503.00075	t	1552
1264	Know What You Don't Know: Unanswerable Questions for SQuAD	P Rajpurkar, R Jia, P Liang	Extractive reading comprehension systems can often locate the correct answer to a question in a context document, but they also tend to make unreliable guesses on questions for which the correct answer is not stated in the context. Existing datasets either focus exclusively on …	\N	2018	https://arxiv.org/abs/1806.03822	f	253
1265	AllenNLP: A deep semantic natural language processing platform	M Gardner, J Grus, M Neumann, O Tafjord…	This paper describes AllenNLP, a platform for research on deep learning methods in natural language understanding. AllenNLP is designed to support researchers who want to build novel language understanding models quickly and easily. It is built on top of PyTorch …	\N	2018	https://arxiv.org/abs/1803.07640	f	195
502	Exploiting similarities among languages for machine translation	T Mikolov, QV Le, I Sutskever	Dictionaries and phrase tables are the basis of modern statistical machine translation systems.\n\nThis paper develops a method that can automate the process of generating and extending dictionaries and phrase tables. Our method can **translate missing word and phrase entries by learning language structures based on large monolingual data and mapping between languages from small bilingual data**.\nIt uses distributed representation of words and learns a **linear mapping between vector spaces of languages**. Despite its simplicity, our method is surprisingly effective: we can achieve almost 90% precision@5 for translation of words between English and Spanish. This method makes little assumption about the languages, so it can be used to extend and refine dictionaries and translation tables for any language pairs.	\N	2013	https://arxiv.org/abs/1309.4168	t	835
2047	A large annotated corpus for learning natural language inference	SR Bowman, G Angeli, C Potts, CD Manning	Understanding entailment and contradiction is fundamental to understanding natural language, and inference about entailment and contradiction is a valuable testing ground for the development of semantic representations. However, machine learning research in this area has been dramatically limited by the lack of large-scale resources.\n\nTo address this, we introduce the **Stanford Natural Language Inference corpus**, a new, freely available collection of labeled sentence pairs, written by humans doing a novel grounded task based on image captioning. At 570K pairs, it is two orders of magnitude larger than all other resources of its type. This increase in scale allows lexicalized classifiers to outperform some sophisticated existing entailment models, and it allows a neural network-based model to perform competitively on natural language inference benchmarks for the first time. 	\N	2015	https://arxiv.org/abs/1508.05326	t	960
1266	Stochastic answer networks for machine reading comprehension	X Liu, Y Shen, K Duh, J Gao	We propose a simple yet robust stochastic answer network (SAN) that simulates multi-step reasoning in machine reading comprehension. Compared to previous work such as ReasoNet which used reinforcement learning to determine the number of steps, the unique …\n\nComment\n----------\n\nDo not themselves use embeddings.	\N	2017	https://arxiv.org/abs/1712.03556	f	77
519	Bilingual word embeddings for phrase-based machine translation	WY Zou, R Socher, D Cer, CD Manning	We introduce bilingual word embeddings: semantic embeddings associated across two languages in the context of neural language models. We propose a method to learn bilingual embeddings from a large unlabeled corpus, while utilizing MT word alignments to …\n\nPaper has very bad quality :(	\N	2013	https://www.aclweb.org/anthology/D13-1141	t	449
2055	Grarep: Learning graph representations with global structural information	S Cao, W Lu, Q Xu	In this paper, we present {GraRep}, a novel model for learning vertex representations of weighted graphs. This model learns low dimensional vectors to represent vertices appearing in a graph and, unlike existing work, integrates global structural information of the graph into …\n\n---\n\nNo use cases.	\N	2015	https://dl.acm.org/citation.cfm?id=2806512	f	547
1202	The natural language decathlon: Multitask learning as question answering	B McCann, NS Keskar, C Xiong, R Socher	Deep learning has improved performance on many natural language processing (NLP) tasks individually. However, general NLP models cannot emerge within a paradigm that focuses on the particularities of a single metric, dataset, and task.\n\nWe introduce the Natural Language Decathlon (decaNLP), a challenge that spans ten tasks: question answering, machine translation, summarization, natural language inference, sentiment analysis, semantic role labeling, zero-shot relation extraction, goal-oriented dialogue, semantic parsing, and commonsense pronoun resolution.\nWe cast all tasks as question answering over a context.\n\nFurthermore, we present a new Multitask Question Answering Network (MQAN) jointly learns all tasks in decaNLP without any task-specific modules or parameters in the multitask setting. MQAN shows improvements in transfer learning for machine translation and named entity recognition, domain adaptation for sentiment analysis and natural language inference, and zero-shot capabilities for text classification. We demonstrate that the MQAN's multi-pointer-generator decoder is key to this success and performance further improves with an anti-curriculum training strategy. Though designed for decaNLP, MQAN also achieves state of the art results on the WikiSQL semantic parsing task in the single-task setting. We also release code for procuring and processing data, training and evaluating models, and reproducing all experiments for decaNLP.	\N	2018	https://arxiv.org/abs/1806.08730	t	83
691	Learning principled bilingual mappings of word embeddings while preserving monolingual invariance	M Artetxe, G Labaka, E Agirre	Mapping word embeddings of different languages into a single space has multiple applications. In order to map from a source space into a target space, a common approach is to learn a linear mapping that minimizes the distances between equivalences listed in a bilingual dictionary.\n\nIn this paper, we **propose a framework that generalizes previous work, provides an efficient exact method to learn the optimal linear transformation and yields the best bilingual results in translation induction** while preserving monolingual performance in an analogy task.	\N	2016	https://www.aclweb.org/anthology/D16-1250	t	161
2451	Advances in pre-training distributed word representations	T Mikolov, E Grave, P Bojanowski, C Puhrsch…	Many Natural Language Processing applications nowadays rely on pre-trained word representations estimated from large text corpora such as news collections, Wikipedia and Web Crawl. In this paper, we show how to train high-quality word vector representations by using a combination of known tricks that are however rarely used together. The main result of our work is the new set of publicly available pre-trained models that outperform the current state of the art by a large margin on a number of tasks. Subjects: Computation and …	\N	2017	https://arxiv.org/abs/1712.09405	t	258
2452	Improving vector space word representations using multilingual correlation	M Faruqui, C Dyer	The distributional hypothesis of Harris (1954), according to which the meaning of words is evidenced by the contexts they occur in, has motivated several effective techniques for obtaining vector space semantic representations of words using unannotated text corpora. This paper argues that lexico-semantic content should additionally be invariant across languages and proposes a simple technique based on canonical correlation analysis (CCA) for incorporating multilingual evidence into vectors generated monolingually. We evaluate …	\N	2014	https://www.aclweb.org/anthology/E14-1049	t	408
521	Multimodal neural language models	R Kiros, R Salakhutdinov, R Zemel	We introduce two multimodal neural language models: models of natural language that can be conditioned on other modalities. An imagetext multimodal neural language model can be used to retrieve images given complex sentence queries, retrieve phrase descriptions given …\n\nComment\n----------\n\nThis is the ancestral version of the Kiros 2014 paper called "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"	\N	2014	http://www.jmlr.org/proceedings/papers/v32/kiros14.pdf	f	463
514	Deep learning applications and challenges in big data analytics	MM Najafabadi, F Villanustre…	Big Data Analytics and Deep Learning are two high-focus of data science. Big Data has become important as many organizations both public and private have been collecting massive amounts of domain-specific information, which can contain useful information about …\n\n----\n\nNo use cases, very shallow mention of WEMs.	\N	2015	https://www.biomedcentral.com/openurl?doi=10.1186/s40537-014-0007-7	f	735
522	Geometric deep learning: going beyond euclidean data	MM Bronstein, J Bruna, Y LeCun…	Geometric deep learning is an umbrella term for emerging techniques attempting to generalize (structured) deep neural models to non-Euclidean domains, such as graphs and manifolds. The purpose of this article is to overview different examples of geometric deep …\n\n	\N	2017	https://ieeexplore.ieee.org/abstract/document/7974879/	f	723
515	A review of relational machine learning for knowledge graphs	M Nickel, K Murphy, V Tresp…	Relational machine learning studies methods for the statistical analysis of relational, or graph-structured, data. In this paper, we provide a review of how such statistical models can be “trained” on large knowledge graphs, and then used to predict new facts about the world …\n\n----\nNo use cases.	\N	2015	https://ieeexplore.ieee.org/abstract/document/7358050/	f	651
516	How to construct deep recurrent neural networks	R Pascanu, C Gulcehre, K Cho, Y Bengio	In this paper, we explore different ways to extend a recurrent neural network (RNN) to a\\textit {deep} RNN. We start by arguing that the concept of depth in an RNN is not as clear as it is in feedforward neural networks. By carefully analyzing and understanding the architecture of …\n\n----\n\nNo use cases.	\N	2013	https://arxiv.org/abs/1312.6026	f	562
2474	Learning word vectors for 157 languages	E Grave, P Bojanowski, P Gupta, A Joulin…	Distributed word representations, or word vectors, have recently been applied to many tasks in natural language processing, leading to state-of-the-art performance. A key ingredient to the successful application of these representations is to train them on very large corpora …	\N	2018	https://arxiv.org/abs/1802.06893	t	215
2472	Tensorflow: Large-scale machine learning on heterogeneous distributed systems	M Abadi, A Agarwal, P Barham, E Brevdo…	TensorFlow is an interface for expressing machine learning algorithms, and an implementation for executing such algorithms. A computation expressed using TensorFlow can be executed with little or no change on a wide variety of heterogeneous systems …\n\nComment\n----------\n\nNo use cases.	\N	2016	https://arxiv.org/abs/1603.04467	f	4606
2473	Layer normalization	JL Ba, JR Kiros, GE Hinton	Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron …\n\nComment\n----------\n\nMentions WEMs (skip-though vectors by Kiros et al.), but names no use cases.	\N	2016	https://arxiv.org/abs/1607.06450	f	1342
2475	Graph neural networks: A review of methods and applications	J Zhou, G Cui, Z Zhang, C Yang, Z Liu, L Wang…	Lots of learning tasks require dealing with graph data which contains rich relation information among elements. Modeling physics system, learning molecular fingerprints, predicting protein interface, and classifying diseases require a model to learn from graph …	\N	2018	https://arxiv.org/abs/1812.08434	t	158
2476	The united nations parallel corpus v1. 0	M Ziemski, M Junczys-Dowmunt…	This paper describes the creation process and statistics of the official United Nations Parallel Corpus, the first parallel corpus composed from United Nations documents published by the original data creator. The parallel corpus presented consists of manually translated UN …	\N	2016	https://www.aclweb.org/anthology/L16-1561.pdf	t	110
2477	CBNU at TREC 2016 Clinical Decision Support Track.	SH Jo, KS Lee, JW Seol	This paper describes the participation of the CBNU team at the TREC Clinical Decision Support track 2015. We propose a query expansion method based on a clinical semantic knowledge and a topic model. The clinical semantic knowledge is constructed by using …	\N	2016	https://pdfs.semanticscholar.org/ce7d/01857c8b02bd60af26b284bca5c01c8b14ed.pdf	t	94
526	Semantics derived automatically from language corpora contain human-like biases	A Caliskan, JJ Bryson, A Narayanan	Machine learning is a means to derive artificial intelligence by discovering patterns in existing data. Here, we show that applying machine learning to ordinary human language results in human-like semantic biases. We replicated a spectrum of known biases, as …\n\n---\n\nNo use cases.	\N	2017	https://science.sciencemag.org/content/356/6334/183.short	f	579
2478	Reinforced mnemonic reader for machine comprehension	M Hu, Y Peng, X Qiu	Recently, several end-to-end neural models have been proposed for machine comprehension (MC) tasks. Most of these models only capture interactions between the context and the query, and utilize” one-shot prediction” to point the boundary of answer …	\N	2017	https://www.arxiv-vanity.com/papers/1705.02798/	t	86
2479	Breaking nli systems with sentences that require simple lexical inferences	M Glockner, V Shwartz, Y Goldberg	We create a new NLI test set that shows the deficiency of state-of-the-art models in inferences that require lexical and world knowledge. The new examples are simpler than the SNLI test set, containing sentences that differ by at most one word from sentences in the …	\N	2018	https://arxiv.org/abs/1805.02266	t	65
2481	Deep graph infomax	P Veličković, W Fedus, WL Hamilton, P Liò…	We present Deep Graph Infomax (DGI), a general approach for learning node representations within graph-structured data in an unsupervised manner. DGI relies on maximizing mutual information between patch representations and corresponding high-level …	\N	2018	https://arxiv.org/abs/1809.10341	t	61
2487	A hierarchical multi-task approach for learning embeddings from semantic tasks	V Sanh, T Wolf, S Ruder	Much effort has been devoted to evaluate whether multi-task learning can be leveraged to learn rich representations that can be used in various Natural Language Processing (NLP) down-stream applications. However, there is still a lack of understanding of the settings in …	\N	2019	https://www.aaai.org/ojs/index.php/AAAI/article/view/4673	t	28
2489	Spanbert: Improving pre-training by representing and predicting spans	M Joshi, D Chen, Y Liu, DS Weld, L Zettlemoyer…	We present SpanBERT, a pre-training method that is designed to better represent and predict spans of text. Our approach extends BERT by (1) masking contiguous random spans, rather than random tokens, and (2) training the span boundary representations to predict the …	\N	2019	https://arxiv.org/abs/1907.10529	t	25
2488	Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks	J Lu, D Batra, D Parikh, S Lee	We present ViLBERT (short for Vision-and-Language BERT), a model for learning task-agnostic joint representations of image content and natural language. We extend the popular BERT architecture to a multi-modal two-stream model, processing both visual and …	\N	2019	http://papers.nips.cc/paper/8297-vilbert-pretraining-task-agnostic-visiolinguistic-representations-for-vision-and-language-tasks	t	25
2490	Albert: A lite bert for self-supervised learning of language representations	Z Lan, M Chen, S Goodman, K Gimpel…	Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations, longer training times, and …	\N	2019	https://arxiv.org/abs/1909.11942	t	23
2491	Visualbert: A simple and performant baseline for vision and language	LH Li, M Yatskar, D Yin, CJ Hsieh…	We propose VisualBERT, a simple and flexible framework for modeling a broad range of vision-and-language tasks. VisualBERT consists of a stack of Transformer layers that implicitly align elements of an input text and regions in an associated input image with self …	\N	2019	https://arxiv.org/abs/1908.03557	t	22
2580	Finding function in form: Compositional character models for open vocabulary word representation	W Ling, T Luís, L Marujo, RF Astudillo, S Amir…	We introduce a model for constructing vector representations of words by composing characters using bidirectional LSTMs. Relative to traditional word representation models that have independent vectors for each word type, our model requires only a single vector per character type and a fixed set of parameters for the compositional model.\n\nDespite the compactness of this model and, more importantly, the arbitrary nature of the form-function relationship in language, our "composed" word representations yield state-of-the-art results in language modeling and part-of-speech tagging. Benefits over traditional baselines are particularly pronounced in morphologically rich languages (e.g., Turkish).	\N	2015	https://arxiv.org/abs/1508.02096	t	413
2533	From'F'to'A'on the NY Regents Science Exams: An Overview of the Aristo Project	P Clark, O Etzioni, T Khot, BD Mishra…	AI has achieved remarkable mastery over games such as Chess, Go, and Poker, and even Jeopardy, but the rich variety of standardized exams has remained a landmark challenge. Even in 2016, the best AI system achieved merely 59.3% on an 8th Grade science exam …	\N	2019	https://arxiv.org/abs/1909.01958	t	4
491	Skip-thought vectors	R Kiros, Y Zhu, RR Salakhutdinov, R Zemel…	We describe an approach for unsupervised learning of a generic, distributed sentence encoder. Using the continuity of text from books, we train an encoder-decoder model that tries to reconstruct the surrounding sentences of an encoded passage. Sentences that share semantic and syntactic properties are thus mapped to similar vector representations.\n\nWe next introduce a simple vocabulary expansion method to encode words that were not seen as part of training, allowing us to expand our vocabulary to a million words. After training our model, we extract and evaluate our vectors with linear models on 8 tasks: semantic relatedness, paraphrase detection, image-sentence ranking, question-type classification and 4 benchmark sentiment and subjectivity datasets.\n\nThe end result is an off-the-shelf encoder that can produce highly generic sentence representations that are robust and perform well in practice. We will make our encoder publicly available.	\N	2015	http://papers.nips.cc/paper/5950-skip-thought-vectors	t	1382
489	Enriching word vectors with subword information	P Bojanowski, E Grave, A Joulin, T Mikolov	Continuous word representations, trained on large unlabeled corpora are useful for many natural language processing tasks. Popular models that learn such representations ignore the morphology of words, by assigning a distinct vector to each word.\n\nThis is a limitation, especially for languages with large vocabularies and many rare words. In this paper, we propose a new approach based on the skipgram model, where each word is represented as a bag of character n-grams. A vector representation is associated to each character n-gram; words being represented as the sum of these representations.\nOur method is fast, allowing to train models on large corpora quickly and allows us to compute word representations for words that did not appear in the training data.\n\nWe evaluate our word representations on nine different languages, both on word similarity and analogy tasks. By comparing to recently proposed morphological word representations, we show that our vectors achieve state-of-the-art performance on these tasks.	\N	2017	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00051	t	2561
2480	Massively multilingual sentence embeddings for zero-shot cross-lingual transfer and beyond	M Artetxe, H Schwenk	We introduce an architecture to learn joint multilingual sentence representations for 93 languages, belonging to more than 30 different families and written in 28 different scripts. Our system uses a single BiLSTM encoder with a shared byte-pair encoding vocabulary for all languages, which is coupled with an auxiliary decoder and trained on publicly available parallel corpora.\n\nThis enables us to learn a classifier on top of the resulting embeddings using English annotated data only, and transfer it to any of the 93 languages without any modification. Our experiments in cross-lingual natural language inference (XNLI data set), cross-lingual document classification (MLDoc data set), and parallel corpus mining (BUCC data set) show the effectiveness of our approach.\n\nWe also introduce a new test set of aligned sentences in 112 languages, and show that our **sentence embeddings obtain strong results in multilingual similarity search even for low- resource languages**. Our implementation, the pre-trained encoder, and the multilingual test set are available at https://github.com/facebookresearch/LASER.	\N	2019	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00288	t	63
500	Dependency-based word embeddings	O Levy, Y Goldberg	While continuous word embeddings are gaining popularity, current models are based solely on linear contexts. In this work, we **generalize the skip-gram model with negative sampling introduced by Mikolov et al. to include arbitrary contexts**. In particular, we perform experiments with dependency-based contexts, and show that they produce markedly different embeddings. The dependency based embeddings are less topical and exhibit more functional similarity than the original skip-gram embeddings.\n\nComment\n----------\n\nFind out, that different notions of "context" produce very different embeddings.\nUse dependency-parser to learn "similarity" embeddings, i.e. "find words which are syntactically similar".\nHowever, they do not describe how these embeddings could be made useful in a system.\nIntrospect embeddings: Which contexts are activated by a word? This could be a use case for linguistics, they do not discuss this however.\n	\N	2014	https://www.aclweb.org/anthology/P14-2050	f	827
479	GloVe: Global Vectors for Word Representation	J Pennington, R Socher, C Manning	Recent methods for learning vector space representations of words have succeeded in capturing fine-grained semantic and syntactic regularities using vector arithmetic, but the origin of these regularities has remained opaque. We analyze and make explicit the model properties needed for such regularities to emerge in word vectors. The result is a new **global log-bilinear regression model** that combines the advantages of the two major model families in the literature: global matrix factorization and local context window methods.\n\nOur model efficiently leverages statistical information by training only on the nonzero elements in a word-word co-occurrence matrix, rather than on the entire sparse matrix or on individual context windows in a large corpus. The model **produces a vector space with meaningful substructure, as evidenced by its performance of 75 % on a recent word analogy task**.\nIt also outperforms related models on similarity tasks and named entity recognition.	\N	2014	https://www.aclweb.org/anthology/D14-1162	t	11335
510	Simlex-999: Evaluating semantic models with (genuine) similarity estimation	F Hill, R Reichart, A Korhonen	We present SimLex-999, a gold standard resource for evaluating distributional semantic models that improves on existing resources in several important ways.\n\nFirst, in contrast to gold standards such as WordSim-353 and MEN, it **explicitly quantifies similarity rather than association or relatedness** so that pairs of entities that are associated but not actually similar (Freud, psychology) have a low rating.\nWe show that, via this focus on similarity, SimLex-999 **incentivizes the development of models with a different, and arguably wider, range of applications than those which reflect conceptual association**.\n\nSecond, SimLex-999 contains a range of concrete and abstract adjective, noun, and verb pairs, together with an independent rating of concreteness and (free) association strength for each pair. This diversity enables fine-grained analyses of the performance of models on concepts of different types, and consequently greater insight into how architectures can be improved.\n\nFurther, unlike existing gold standard evaluations, for which automatic approaches have reached or surpassed the inter-annotator agreement ceiling, state-of-the-art models perform well below this ceiling on SimLex-999. There is therefore plenty of scope for SimLex-999 to quantify future improvements to distributional semantic models, **guiding the development of the next generation of representation-learning architectures.**\n\n\nComment\n----------\n\nSimlex-999 implements a specific word distance metric: *similarity* (as opposed to *assocation/relatedness*).\n\nFor various WEMs the authors check the correlation between vector distances and Simlex-999 distances and find that current models tend to implement *association* and not *similarity*.	\N	2015	https://www.mitpressjournals.org/doi/abs/10.1162/COLI_a_00237	t	690
591	Word translation without parallel data	A Conneau, G Lample, MA Ranzato, L Denoyer…	State-of-the-art methods for learning cross-lingual word embeddings have relied on bilingual dictionaries or parallel corpora. Recent studies showed that the need for parallel data supervision can be alleviated with character-level information. While these methods showed encouraging results, they are not on par with their supervised counterparts and are limited to pairs of languages sharing a common alphabet.\n\nIn this work, we show that we can build a bilingual dictionary between two languages without using any parallel corpora, by **aligning monolingual word embedding spaces in an unsupervised way**. Without using any character information, our model even outperforms existing supervised methods on cross-lingual tasks for some language pairs.\n\nOur experiments demonstrate that our method works very well also for distant language pairs, like English-Russian or English-Chinese. We finally describe experiments on the English-Esperanto low-resource language pair, on which there only exists a limited amount of parallel data, to show the potential impact of our method in fully unsupervised machine translation. Our code, embeddings and dictionaries are publicly available. 	\N	2017	https://arxiv.org/abs/1710.04087	t	345
509	Ask me anything: Dynamic memory networks for natural language processing	A Kumar, O Irsoy, P Ondruska, M Iyyer…	Most tasks in natural language processing can be cast into question answering (QA) problems over language input. We introduce the dynamic memory network (DMN), a neural network architecture which processes input sequences and questions, forms episodic memories, and generates relevant answers.\nQuestions trigger an iterative attention process which allows the model to condition its attention on the inputs and the result of previous iterations.\nThese results are then reasoned over in a hierarchical recurrent sequence model to generate answers.\n\nThe DMN can be trained end-to-end and obtains state-of-the-art results on several types of tasks and datasets: question answering (Facebook's bAbI dataset), text classification for sentiment analysis (Stanford Sentiment Treebank) and sequence modeling for part-of-speech tagging (WSJ-PTB).\nThe training for these different tasks **relies exclusively on trained word vector representations and input-question-answer triplets**. 	\N	2016	http://www.jmlr.org/proceedings/papers/v48/kumar16.pdf	t	718
511	Grammar as a foreign language	O Vinyals, Ł Kaiser, T Koo, S Petrov…	Syntactic constituency parsing is a fundamental problem in natural language processing and has been the subject of intensive research and engineering for decades. As a result, the most accurate parsers are domain specific, complex, and inefficient.\n\nIn this paper we show that the domain agnostic attention-enhanced sequence-to-sequence model achieves state-of-the-art results on the most widely used syntactic constituency parsing dataset, when trained on a large synthetic corpus that was annotated using existing parsers.\nIt also matches the performance of standard parsers when trained only on a small human-annotated dataset, which shows that this model is highly data-efficient, in contrast to sequence-to-sequence models without the attention mechanism.\nOur parser is also fast, processing over a hundred sentences per second with an unoptimized CPU implementation. 	\N	2015	http://papers.nips.cc/paper/5635-grammar-as-a-foreign-language	t	636
520	A primer on neural network models for natural language processing	Y Goldberg	Over the past few years, neural networks have re-emerged as powerful machine-learning models, yielding state-of-the-art results in fields such as image recognition and speech processing.\n\nMore recently, neural network models started to be applied also to textual natural language signals, again with very promising results. This tutorial surveys neural network models from the perspective of natural language processing research, in an attempt to bring natural-language researchers up to speed with the neural techniques. The tutorial covers input encoding for natural language tasks, feed-forward networks, convolutional networks, recurrent networks and recursive networks, as well as the computation graph abstraction for automatic gradient computation. \n\n---\n\nNo use cases.	\N	2016	http://www.jair.org/papers/paper4992.html	f	547
2484	Unsupervised data augmentation	Q Xie, Z Dai, E Hovy, MT Luong, QV Le	Semi-supervised learning lately has shown much promise in improving deep learning models when labeled data is scarce.\nCommon among recent approaches is the use of consistency training on a large amount of unlabeled data to constrain model predictions to be invariant to input noise.\n\nIn this work, we present a new perspective on how to effectively noise unlabeled examples and argue that the **quality of noising, specifically those produced by advanced data augmentation methods, plays a crucial role in semi-supervised learning**. By substituting simple noising operations with advanced data augmentation methods, our method brings substantial improvements across six language and three vision tasks under the same consistency training framework.\n\nOn the IMDb text classification dataset, with only 20 labeled examples, our method achieves an error rate of 4.20, outperforming the state-of-the-art model trained on 25,000 labeled examples.\nOn a standard semi-supervised learning benchmark, CIFAR-10, our method outperforms all previous approaches and achieves an error rate of 2.7% with only 4,000 examples, nearly matching the performance of models trained on 50,000 labeled examples.\n\nOur method also combines well with transfer learning, e.g., when finetuning from BERT, and yields improvements in high-data regime, such as ImageNet, whether when there is only 10% labeled data or when a full labeled set with 1.3M extra unlabeled examples is used.\n\nComment\n----------\n\nNo use cases of WEMs. Just training details.	\N	2019	https://arxiv.org/abs/1904.12848	f	48
705	Many languages, one parser	W Ammar, G Mulcaire, M Ballesteros, C Dyer…	We train one multilingual model for dependency parsing and use it to parse sentences in several languages. The parsing model uses\n\n 1. multilingual word clusters and embeddings;\n 2. token-level language information; and\n 3.  language-specific features (fine-grained POS tags).\n\nThis input representation enables the parser not only to parse effectively in multiple languages, but also to **generalize across languages based on linguistic universals and typological similarities, making it more effective to learn from limited annotations**.\nOur parser’s performance compares favorably to strong baselines in a range of data scenarios, including when the target language has a large treebank, a small treebank, or no treebank for training.\n	\N	2016	https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00109	t	114
2486	Unsupervised word embeddings capture latent knowledge from materials science literature	V Tshitoyan, J Dagdelen, L Weston, A Dunn, Z Rong…	The overwhelming majority of scientific knowledge is published as text, which is difficult to analyse by either traditional statistical analysis or modern machine learning methods. By contrast, the main source of machine-interpretable data for the materials research community has come from structured property databases which encompass only a small fraction of the knowledge present in the research literature.\n\nBeyond property values, publications contain valuable knowledge regarding the connections and relationships between data items as interpreted by the authors.\nTo improve the identification and use of this knowledge, several studies have focused on the retrieval of information from scientific literature using supervised natural language processing which requires large hand-labelled datasets for training.\n\nHere we show that materials science knowledge present in the published literature can be efficiently encoded as information-dense word embeddings (vector representations of words) without human labelling or supervision.\nWithout any explicit insertion of chemical knowledge, these **embeddings capture complex materials science concepts such as the underlying structure of the periodic table** and structure–property relationships in materials.\n\nFurthermore, we demonstrate that an **unsupervised method can recommend materials for functional applications several years before their discovery**.\nThis suggests that **latent knowledge regarding future discoveries is to a large extent embedded in past publications**.\nOur findings highlight the possibility of extracting knowledge and relationships from the massive body of scientific literature in a collective manner, and point towards a generalized approach to the mining of scientific literature.	\N	2019	https://www.nature.com/articles/s41586-019-1335-8	t	40
\.


--
-- Data for Name: use_case_mention_models; Type: TABLE DATA; Schema: public; Owner: taxonomist
--

COPY public.use_case_mention_models (mention_id, mention_model) FROM stdin;
34	3
44	3
35	3
45	3
48	6
41	3
37	3
54	3
46	1
49	6
38	3
57	3
36	3
53	6
50	6
42	5
58	1
56	6
52	6
60	6
61	1
62	1
64	3
63	6
39	3
77	3
69	10
74	3
65	3
67	11
72	13
71	11
68	9
70	3
78	3
43	12
73	13
66	6
75	14
47	16
81	3
76	14
79	17
80	18
82	6
82	1
36	19
69	3
58	6
83	3
84	6
85	6
86	21
87	20
88	20
64	23
43	3
43	23
47	3
89	6
90	19
90	3
91	3
92	24
93	3
94	3
\.


--
-- Data for Name: use_case_mentions; Type: TABLE DATA; Schema: public; Owner: taxonomist
--

COPY public.use_case_mentions (mention_use_case, mention_publication, mention_description, mention_id) FROM stdin;
27	483	Use skip-gram embeddings to train a LSTM model generating *complete sentences* to describe images.\n\n`Embedding(Horse)` is close to `Embedding(Unicorn)`, therefore two pictures showing a horse and a unicorn in the training set will produce similar embedding inputs and therefore 'encourage' the CNN to treat both kinds of images the same way.	44
29	492	Uses skip-gram embeddings to improve a neural network generating captions for images.\n\nLower layers of the neural network are retrained to predict embeddings of the class labels instead of a class-label distribution.\nk-NN search then yields the predicted class labels for each predicted embedding.\n\nMakes "**more semantically reasonable errors** and [...] the semantic information can be exploited to **make predictions about tens of thousands of image labels not observed during training**"	35
27	2045	Pre-trained GloVe embeddings are used to represent words, which are then fed into a **Tree-LSTM** as inputs for various tasks.	48
32	523		41
27	1342	Embeddings are learned for all distinct values in the database (= words). Enhance the input to a NN for query optimization.	54
27	1794	ELMo embeddings are added to the individual models inputs.	46
27	2047	Pre-trained GloVe embeddings are used to enhance the inputs of a neural net.	49
30	652		38
29	2057	Class labels are embedded.	57
27	2051	GloVe embeddings as input features for a NN.	53
27	2049	GloVe embeddings enhance the input of a LSTM.	50
27	1267		58
27	2053	Embeddings enhance the input.	56
27	2050	GloVe embeddings enhance the input of a neural network.	52
27	1202	GloVe embeddings enhance the input to a multi-task question answering model (MQAN).	60
27	1273		61
27	1270		62
29	2051	NN learns to predict GloVe-based embeddings.	63
30	691	Mapping is learned to translate between two languages.	77
27	524	Character embeddings + skip-gram embeddings enhance the input for a NN.	69
31	510	This paper measures how well distances between embeddings of a WEM can be interpreted as dissimilarity scores.	74
27	511	Use pre-trained skip-gram embeddings as input to neural network.	65
27	519	Authors train embeddings with a bilingual alignment objective, based on Collobert embeddings.\nThese aligned embeddings are better than the original ones.	67
31	518	Various forms of word similarities are computed as vector distances.	72
31	519	Distances are used to estimate semantic similarity.	71
27	531	Sentence embeddings enhance the input for many tasks.	68
31	525	The noise-contrastive embeddings permit interpretation of distances.	70
30	591	A linear mapping between two vector spaces is learned **unsupervised**.	78
31	506	Distances are interpreted as semantic dissimilarity.	42
27	482	CBOW and skip-gram are extended to form *paragraph embeddings* (PEM).	43
27	518	Embeddings are used as enhanced input to ML model.	73
27	2054	Skip-grams and GloVe embeddings are used as inputs to a CNN.	66
27	705	Multilingual embeddings allow to implement a multilingual parser.	75
31	479	Distances are used to solve Mikolovs analogy task.	84
30	502	The vector space of the skip-gram model is used for the mapping.	37
31	2486	Authors train skip-grams on text from materials science.	81
27	2487	Embeddings enhance the input.	82
27	748	Embeddings enhance the input of a ML model for dependency parsing.	76
31	507	Distances are interpreted as semantic dissimilarity.	45
27	491	The input of several **linear** models is improved by the skip-thought vectors.	88
27	479	GloVe embeddings enhance the input of ML models.	85
31	2452	Bilingual word vectors perform better in similarity benchmarks.	79
27	494	Skip-gram embeddings and character-based embeddings are used as inputs.	36
27	545	Skip-grams enhance the input.	94
31	491	Distances (dot-product and absolute distance) are used to solve a paraphrase detection task.	87
31	1805	Distances are used to solve analogical reasoning tasks.	64
31	529	Cosine distance between sentence embeddings is interpreted as semantic dissimilarity.	92
27	2044	WIEMs (word-image embeddings) are used as inputs for a RNN in various tasks.\nWord embeddings are initialized with skip-grams.	47
31	476	Distances are used to solve analogical reasoning tasks.	83
31	2480	Sentence embeddings capture the semantic similarity of the sentences well.	80
31	489	Distances are used to solve word similarity and analogy tasks.	86
27	527	Skip-gram embeddings are used in one of the models in the same way as in [Vinyals 2015].	91
27	501	Skip-gram embeddings enhance the input of the model to solve tasks.	34
27	509	Embeddings serve as inputs for complex task models.	89
27	2580	Character-based embeddings enhance the input for various task-specific models.	90
31	505	Word distances are interpreted as semantic dissimilarity.	39
31	528	Different definitions for distance metrics are evaluated. Some of them can better be interpreted as semantic similarity than others.	93
\.


--
-- Data for Name: use_cases; Type: TABLE DATA; Schema: public; Owner: taxonomist
--

COPY public.use_cases (uc_description, uc_id, uc_title, uc_short) FROM stdin;
A new model can be trained to **predict the embeddings of the original outputs**.\nThe mapping from outputs to embeddings must be bidirectional to\n\n 1. embed the original outputs into the vector space before training, and\n 2. retrieve the desired outputs from the predicted embeddings during prediction.\n\nDuring training, the model will be shown similar embeddings for semantically related outputs.\nIdeally, this **encourages the model to discriminate its inputs according to semantical differences**, i.e. if two inputs map to similar embeddings it will focus on their commonalities and if not, it will focus on their differences.\n\nFurthermore, the enhanced model will also be able to perform **zero-shot learning**.	29	Learning Embeddings	TL.LearnEmbed
A WEM can be used for *Feature engineering* in a machine learning model, by mapping existing features to embeddings before training.\n\nOften the embeddings are provided additionally to the original input instead of replacing it.\n\nThe embeddings **inform the new model about the semantic and syntactic aspects of the original input**, which is often relevant to the mapping to be learned (especially for NLP problems).	27	Enhancing the Input	TL.EnhanceIn
The WEM is treated like an index, which is queried for additional\nsemantic information about the original model output to enhance the final output.\nThe **original model can be reused without any additional training.** But: There has to be a natural bidirectional mapping from model outputs to embeddings.\n\nCan be used for *Zero-shot learning*:\nThe mapping from embeddings to the final outputs can produce semantically and syntactically sensible outputs which were never learned by the original model.	32	Enhancing the Output	TL.EnhanceOut
Given two vector spaces of different WEMs, there might be a simple natural mapping between the two, translating embeddings from one WEM into embeddings of the other.\n\nProvided that the individual WEMs are meaningful in their respective domains, such a **mapping can serve as a translation between the vocabularies of these domains.**	30	Mapping Between Word Vector Spaces	IP.VecMap
Semantic and syntactical relations between words can be evaluated in the vector space produced by a WEM using distance vectors (Mikolov 2013).\n\nSuch distance vectors can e.g. be used to **improve the ability of systems to induce outputs by semantical analogy.**	31	Interpreting Distances of Word Vectors	IP.VecDist
\.


--
-- Name: domain_applications_app_id_seq; Type: SEQUENCE SET; Schema: public; Owner: taxonomist
--

SELECT pg_catalog.setval('public.domain_applications_app_id_seq', 83, true);


--
-- Name: domain_mentions_dmention_id_seq; Type: SEQUENCE SET; Schema: public; Owner: taxonomist
--

SELECT pg_catalog.setval('public.domain_mentions_dmention_id_seq', 1, false);


--
-- Name: domains_dom_id_seq; Type: SEQUENCE SET; Schema: public; Owner: taxonomist
--

SELECT pg_catalog.setval('public.domains_dom_id_seq', 29, true);


--
-- Name: model_types_id_seq; Type: SEQUENCE SET; Schema: public; Owner: taxonomist
--

SELECT pg_catalog.setval('public.model_types_id_seq', 24, true);


--
-- Name: origins_origin_id_seq; Type: SEQUENCE SET; Schema: public; Owner: taxonomist
--

SELECT pg_catalog.setval('public.origins_origin_id_seq', 61, true);


--
-- Name: publication_id_seq; Type: SEQUENCE SET; Schema: public; Owner: taxonomist
--

SELECT pg_catalog.setval('public.publication_id_seq', 2706, true);


--
-- Name: use_case_mention_mention_id_seq; Type: SEQUENCE SET; Schema: public; Owner: taxonomist
--

SELECT pg_catalog.setval('public.use_case_mention_mention_id_seq', 94, true);


--
-- Name: use_cases_uc_id_seq; Type: SEQUENCE SET; Schema: public; Owner: taxonomist
--

SELECT pg_catalog.setval('public.use_cases_uc_id_seq', 32, true);


--
-- Name: domain_applications domain_applications_pk; Type: CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.domain_applications
    ADD CONSTRAINT domain_applications_pk PRIMARY KEY (app_id);


--
-- Name: domains domains_pkey; Type: CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.domains
    ADD CONSTRAINT domains_pkey PRIMARY KEY (dom_id);


--
-- Name: models models_pk; Type: CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT models_pk PRIMARY KEY (model_id);


--
-- Name: origins origins_pkey; Type: CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.origins
    ADD CONSTRAINT origins_pkey PRIMARY KEY (origin_id);


--
-- Name: publications publication_pkey; Type: CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.publications
    ADD CONSTRAINT publication_pkey PRIMARY KEY (pub_id);


--
-- Name: use_case_mentions use_case_mention_pk; Type: CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.use_case_mentions
    ADD CONSTRAINT use_case_mention_pk PRIMARY KEY (mention_id);


--
-- Name: use_cases use_cases_pk; Type: CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.use_cases
    ADD CONSTRAINT use_cases_pk PRIMARY KEY (uc_id);


--
-- Name: domain_applications_app_id_uindex; Type: INDEX; Schema: public; Owner: taxonomist
--

CREATE UNIQUE INDEX domain_applications_app_id_uindex ON public.domain_applications USING btree (app_id);


--
-- Name: domains_dom_id_uindex; Type: INDEX; Schema: public; Owner: taxonomist
--

CREATE UNIQUE INDEX domains_dom_id_uindex ON public.domains USING btree (dom_id);


--
-- Name: domains_dom_name_uindex; Type: INDEX; Schema: public; Owner: taxonomist
--

CREATE UNIQUE INDEX domains_dom_name_uindex ON public.domains USING btree (dom_name);


--
-- Name: models_id_uindex; Type: INDEX; Schema: public; Owner: taxonomist
--

CREATE UNIQUE INDEX models_id_uindex ON public.models USING btree (model_id);


--
-- Name: domain_applications domain_mentions_domains_dom_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.domain_applications
    ADD CONSTRAINT domain_mentions_domains_dom_id_fk FOREIGN KEY (app_domain) REFERENCES public.domains(dom_id);


--
-- Name: domain_applications domain_mentions_use_case_mention_mention_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.domain_applications
    ADD CONSTRAINT domain_mentions_use_case_mention_mention_id_fk FOREIGN KEY (app_use_case_mention) REFERENCES public.use_case_mentions(mention_id);


--
-- Name: models models_publications_pub_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT models_publications_pub_id_fk FOREIGN KEY (model_publication) REFERENCES public.publications(pub_id);


--
-- Name: origins origins_publications_pub_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.origins
    ADD CONSTRAINT origins_publications_pub_id_fk FOREIGN KEY (origin_cites) REFERENCES public.publications(pub_id);


--
-- Name: publication_origins publication_origins_origins_origin_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.publication_origins
    ADD CONSTRAINT publication_origins_origins_origin_id_fk FOREIGN KEY (pub_origin) REFERENCES public.origins(origin_id);


--
-- Name: publication_origins publication_origins_publications_pub_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.publication_origins
    ADD CONSTRAINT publication_origins_publications_pub_id_fk FOREIGN KEY (pub_id) REFERENCES public.publications(pub_id);


--
-- Name: use_case_mention_models use_case_mention_models_models_model_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.use_case_mention_models
    ADD CONSTRAINT use_case_mention_models_models_model_id_fk FOREIGN KEY (mention_model) REFERENCES public.models(model_id);


--
-- Name: use_case_mention_models use_case_mention_models_use_case_mentions_mention_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.use_case_mention_models
    ADD CONSTRAINT use_case_mention_models_use_case_mentions_mention_id_fk FOREIGN KEY (mention_id) REFERENCES public.use_case_mentions(mention_id);


--
-- Name: use_case_mentions use_case_mention_use_cases_uc_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.use_case_mentions
    ADD CONSTRAINT use_case_mention_use_cases_uc_id_fk FOREIGN KEY (mention_use_case) REFERENCES public.use_cases(uc_id);


--
-- Name: use_case_mentions use_cases_publications_publications_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: taxonomist
--

ALTER TABLE ONLY public.use_case_mentions
    ADD CONSTRAINT use_cases_publications_publications_id_fk FOREIGN KEY (mention_publication) REFERENCES public.publications(pub_id);


--
-- PostgreSQL database dump complete
--

