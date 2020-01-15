--
-- PostgreSQL database dump
--

-- Dumped from database version 11.5
-- Dumped by pg_dump version 11.5

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

