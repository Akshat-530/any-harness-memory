--
-- PostgreSQL database dump
--


-- Dumped from database version 17.11 (Debian 17.11-1.pgdg13+2)
-- Dumped by pg_dump version 17.11 (Debian 17.11-1.pgdg13+2)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


--
-- Name: cost_telemetry; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cost_telemetry (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    operation text NOT NULL,
    model text NOT NULL,
    provider text,
    tokens bigint DEFAULT '0'::bigint NOT NULL,
    cost_usd numeric(10,6),
    latency_ms integer,
    success boolean DEFAULT true NOT NULL,
    error text,
    session_id uuid,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: identities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.identities (
    token_digest text NOT NULL,
    identity text NOT NULL,
    scopes text DEFAULT ''::text NOT NULL,
    active boolean DEFAULT true NOT NULL,
    note text DEFAULT ''::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    last_used_at timestamp with time zone
);


--
-- Name: maintenance_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.maintenance_log (
    id bigint NOT NULL,
    ts timestamp with time zone DEFAULT now() NOT NULL,
    run_id text,
    step text NOT NULL,
    action text NOT NULL,
    memory_id text,
    before jsonb,
    after jsonb,
    reason text DEFAULT ''::text NOT NULL,
    dry_run boolean DEFAULT true NOT NULL
);


--
-- Name: maintenance_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.maintenance_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: maintenance_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.maintenance_log_id_seq OWNED BY public.maintenance_log.id;


--
-- Name: mem_audit; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_audit (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    memory_id text,
    scope text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    op text,
    detail jsonb,
    actor text
);


--
-- Name: mem_canonical; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_canonical (
    memory_id text NOT NULL,
    scope text NOT NULL,
    node_type text DEFAULT 'fact'::text NOT NULL,
    status text DEFAULT 'pending_index'::text NOT NULL,
    version bigint DEFAULT '1'::bigint NOT NULL,
    content text,
    valid_from timestamp with time zone DEFAULT now() NOT NULL,
    valid_until timestamp with time zone,
    superseded_by text,
    deleted_at timestamp with time zone,
    last_error text,
    payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT mem_canonical_status_ck CHECK ((status = ANY (ARRAY['pending_index'::text, 'active'::text, 'index_failed'::text, 'superseded'::text, 'deleted'::text])))
);

ALTER TABLE ONLY public.mem_canonical FORCE ROW LEVEL SECURITY;


--
-- Name: mem_edges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_edges (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    src uuid NOT NULL,
    dst uuid NOT NULL,
    rel text NOT NULL,
    confidence real DEFAULT '1'::real NOT NULL,
    valid_from timestamp with time zone DEFAULT now() NOT NULL,
    valid_to timestamp with time zone,
    payload jsonb DEFAULT '{}'::jsonb NOT NULL
);


--
-- Name: mem_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_events (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    memory_id uuid NOT NULL,
    scope text DEFAULT 'global'::text NOT NULL,
    event_type text DEFAULT 'episodic'::text NOT NULL,
    task_signature text,
    payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    occurred_at timestamp with time zone DEFAULT now() NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: mem_feedback; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_feedback (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    memory_id text NOT NULL,
    rating text NOT NULL,
    query text,
    task_type text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    query_hash text
);


--
-- Name: mem_nodes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_nodes (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    memory_id text,
    scope text,
    payload jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    label text,
    node_type text,
    valid_from timestamp with time zone,
    valid_to timestamp with time zone,
    superseded_by uuid
);


--
-- Name: mem_outbox; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_outbox (
    id bigint NOT NULL,
    event_id text NOT NULL,
    event_type text NOT NULL,
    memory_id text NOT NULL,
    scope text NOT NULL,
    version bigint NOT NULL,
    payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    status text DEFAULT 'pending'::text NOT NULL,
    attempts integer DEFAULT 0 NOT NULL,
    last_error text,
    available_at timestamp with time zone DEFAULT now() NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    processed_at timestamp with time zone,
    worker_id text,
    last_worker text,
    lease_expires_at timestamp with time zone,
    first_attempt_at timestamp with time zone,
    last_attempt_at timestamp with time zone,
    next_attempt_at timestamp with time zone,
    dead_letter_reason text,
    applied_version bigint,
    outcome text
);


--
-- Name: mem_outbox_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.mem_outbox ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.mem_outbox_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: mem_procedures; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_procedures (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    memory_id text,
    task_signature text,
    learned_from_outcome text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    scope text,
    proc_name text,
    importance integer,
    source_trajectory text,
    steps jsonb,
    tags text,
    valid_from timestamp with time zone,
    valid_to timestamp with time zone,
    superseded_by uuid,
    payload jsonb
);


--
-- Name: mem_trajectories; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_trajectories (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    task text,
    task_type text,
    outcome text NOT NULL,
    steps_json jsonb DEFAULT '[]'::jsonb NOT NULL,
    failed_step_index integer,
    step_count integer DEFAULT 0 NOT NULL,
    success_count integer DEFAULT 0 NOT NULL,
    failure_count integer DEFAULT 0 NOT NULL,
    duration_minutes integer,
    lessons_learned jsonb,
    facts_discovered jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    task_signature text,
    duration_s double precision
);


--
-- Name: mem_trajectory_steps; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_trajectory_steps (
    id bigint NOT NULL,
    trajectory_id uuid NOT NULL,
    step_index integer NOT NULL,
    action text NOT NULL,
    observation text,
    success boolean DEFAULT true NOT NULL,
    duration_ms integer,
    tool_used text,
    logged_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: mem_trajectory_steps_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mem_trajectory_steps_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: mem_trajectory_steps_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mem_trajectory_steps_id_seq OWNED BY public.mem_trajectory_steps.id;


--
-- Name: op_telemetry; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.op_telemetry (
    id bigint NOT NULL,
    ts timestamp with time zone DEFAULT now() NOT NULL,
    tool text NOT NULL,
    duration_ms integer,
    status text DEFAULT 'ok'::text NOT NULL,
    scope text DEFAULT 'global'::text NOT NULL,
    caller text,
    quality_score real
);


--
-- Name: op_telemetry_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.op_telemetry_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: op_telemetry_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.op_telemetry_id_seq OWNED BY public.op_telemetry.id;


--
-- Name: sessions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sessions (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    token_hash text NOT NULL,
    scope text DEFAULT 'global'::text NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    last_used_at timestamp with time zone,
    user_agent text,
    ip_address text
);


--
-- Name: task_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.task_runs (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    task_signature text NOT NULL,
    task_type text,
    outcome text NOT NULL,
    steps_json jsonb DEFAULT '[]'::jsonb NOT NULL,
    failed_step_index integer,
    step_count integer DEFAULT 0 NOT NULL,
    success_count integer DEFAULT 0 NOT NULL,
    failure_count integer DEFAULT 0 NOT NULL,
    duration_minutes integer,
    lessons_learned jsonb,
    facts_discovered jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone
);


--
-- Name: token_telemetry; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.token_telemetry (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    model text NOT NULL,
    provider text,
    prompt_tokens bigint DEFAULT '0'::bigint NOT NULL,
    completion_tokens bigint DEFAULT '0'::bigint NOT NULL,
    total_tokens bigint DEFAULT '0'::bigint NOT NULL,
    cost_usd numeric(10,6),
    task_type text,
    task_signature text,
    session_id uuid,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    username text NOT NULL,
    email text,
    password_hash text,
    display_name text,
    is_active boolean DEFAULT true NOT NULL,
    is_admin boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    last_login timestamp with time zone
);


--
-- Name: maintenance_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.maintenance_log ALTER COLUMN id SET DEFAULT nextval('public.maintenance_log_id_seq'::regclass);


--
-- Name: mem_trajectory_steps id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_trajectory_steps ALTER COLUMN id SET DEFAULT nextval('public.mem_trajectory_steps_id_seq'::regclass);


--
-- Name: op_telemetry id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.op_telemetry ALTER COLUMN id SET DEFAULT nextval('public.op_telemetry_id_seq'::regclass);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: cost_telemetry cost_telemetry_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cost_telemetry
    ADD CONSTRAINT cost_telemetry_pkey PRIMARY KEY (id);


--
-- Name: identities identities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.identities
    ADD CONSTRAINT identities_pkey PRIMARY KEY (token_digest);


--
-- Name: maintenance_log maintenance_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.maintenance_log
    ADD CONSTRAINT maintenance_log_pkey PRIMARY KEY (id);


--
-- Name: mem_audit mem_audit_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_audit
    ADD CONSTRAINT mem_audit_pkey PRIMARY KEY (id);


--
-- Name: mem_canonical mem_canonical_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_canonical
    ADD CONSTRAINT mem_canonical_pkey PRIMARY KEY (memory_id);


--
-- Name: mem_edges mem_edges_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_edges
    ADD CONSTRAINT mem_edges_pkey PRIMARY KEY (id);


--
-- Name: mem_events mem_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_events
    ADD CONSTRAINT mem_events_pkey PRIMARY KEY (id);


--
-- Name: mem_feedback mem_feedback_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_feedback
    ADD CONSTRAINT mem_feedback_pkey PRIMARY KEY (id);


--
-- Name: mem_nodes mem_nodes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_nodes
    ADD CONSTRAINT mem_nodes_pkey PRIMARY KEY (id);


--
-- Name: mem_outbox mem_outbox_event_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_outbox
    ADD CONSTRAINT mem_outbox_event_id_key UNIQUE (event_id);


--
-- Name: mem_outbox mem_outbox_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_outbox
    ADD CONSTRAINT mem_outbox_pkey PRIMARY KEY (id);


--
-- Name: mem_procedures mem_procedures_memory_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_procedures
    ADD CONSTRAINT mem_procedures_memory_id_key UNIQUE (memory_id);


--
-- Name: mem_procedures mem_procedures_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_procedures
    ADD CONSTRAINT mem_procedures_pkey PRIMARY KEY (id);


--
-- Name: mem_trajectories mem_trajectories_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_trajectories
    ADD CONSTRAINT mem_trajectories_pkey PRIMARY KEY (id);


--
-- Name: mem_trajectory_steps mem_trajectory_steps_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_trajectory_steps
    ADD CONSTRAINT mem_trajectory_steps_pkey PRIMARY KEY (id);


--
-- Name: op_telemetry op_telemetry_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.op_telemetry
    ADD CONSTRAINT op_telemetry_pkey PRIMARY KEY (id);


--
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- Name: sessions sessions_token_hash_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_token_hash_key UNIQUE (token_hash);


--
-- Name: task_runs task_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.task_runs
    ADD CONSTRAINT task_runs_pkey PRIMARY KEY (id);


--
-- Name: token_telemetry token_telemetry_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.token_telemetry
    ADD CONSTRAINT token_telemetry_pkey PRIMARY KEY (id);


--
-- Name: users users_email_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_key UNIQUE (email);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: cost_telemetry_created_at_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX cost_telemetry_created_at_idx ON public.cost_telemetry USING btree (created_at);


--
-- Name: cost_telemetry_model_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX cost_telemetry_model_idx ON public.cost_telemetry USING btree (model);


--
-- Name: cost_telemetry_operation_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX cost_telemetry_operation_idx ON public.cost_telemetry USING btree (operation);


--
-- Name: identities_identity_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX identities_identity_idx ON public.identities USING btree (identity);


--
-- Name: maintenance_log_step_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX maintenance_log_step_idx ON public.maintenance_log USING btree (step, ts);


--
-- Name: maintenance_log_ts_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX maintenance_log_ts_idx ON public.maintenance_log USING btree (ts);


--
-- Name: mem_audit_created_at_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_audit_created_at_idx ON public.mem_audit USING btree (created_at);


--
-- Name: mem_audit_memory_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_audit_memory_id_idx ON public.mem_audit USING btree (memory_id);


--
-- Name: mem_audit_scope; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_audit_scope ON public.mem_audit USING btree (scope, created_at);


--
-- Name: mem_canonical_scope_status_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_canonical_scope_status_idx ON public.mem_canonical USING btree (scope, status);


--
-- Name: mem_canonical_status_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_canonical_status_idx ON public.mem_canonical USING btree (status);


--
-- Name: mem_edges_sdr_uniq; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX mem_edges_sdr_uniq ON public.mem_edges USING btree (src, dst, rel);


--
-- Name: mem_edges_src_dst; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_edges_src_dst ON public.mem_edges USING btree (src, dst, rel);


--
-- Name: mem_events_mem; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_events_mem ON public.mem_events USING btree (memory_id);


--
-- Name: mem_events_task; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_events_task ON public.mem_events USING btree (task_signature, occurred_at DESC);


--
-- Name: mem_events_time; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_events_time ON public.mem_events USING btree (scope, occurred_at DESC);


--
-- Name: mem_feedback_memory_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_feedback_memory_id_idx ON public.mem_feedback USING btree (memory_id);


--
-- Name: mem_nodes_memory_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_nodes_memory_id_idx ON public.mem_nodes USING btree (memory_id);


--
-- Name: mem_nodes_scope_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_nodes_scope_idx ON public.mem_nodes USING btree (scope);


--
-- Name: mem_outbox_claim_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_outbox_claim_idx ON public.mem_outbox USING btree (status, available_at);


--
-- Name: mem_outbox_lease_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_outbox_lease_idx ON public.mem_outbox USING btree (status, next_attempt_at, lease_expires_at);


--
-- Name: mem_outbox_memory_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_outbox_memory_idx ON public.mem_outbox USING btree (memory_id);


--
-- Name: mem_outbox_worker_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_outbox_worker_idx ON public.mem_outbox USING btree (worker_id, status);


--
-- Name: mem_procedures_task_signature_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_procedures_task_signature_idx ON public.mem_procedures USING btree (task_signature);


--
-- Name: mem_trajectories_created_at_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_trajectories_created_at_idx ON public.mem_trajectories USING btree (created_at);


--
-- Name: mem_trajectories_outcome_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_trajectories_outcome_idx ON public.mem_trajectories USING btree (outcome);


--
-- Name: mem_trajectories_task_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_trajectories_task_idx ON public.mem_trajectories USING btree (task);


--
-- Name: mem_trajectories_task_signature_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_trajectories_task_signature_idx ON public.mem_trajectories USING btree (task_signature);


--
-- Name: mem_trajectory_steps_tool_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_trajectory_steps_tool_idx ON public.mem_trajectory_steps USING btree (tool_used);


--
-- Name: mem_trajectory_steps_traj_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX mem_trajectory_steps_traj_idx ON public.mem_trajectory_steps USING btree (trajectory_id, step_index);


--
-- Name: op_telemetry_tool_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX op_telemetry_tool_idx ON public.op_telemetry USING btree (tool, ts);


--
-- Name: op_telemetry_ts_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX op_telemetry_ts_idx ON public.op_telemetry USING btree (ts);


--
-- Name: sessions_expires_at_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX sessions_expires_at_idx ON public.sessions USING btree (expires_at);


--
-- Name: sessions_token_hash_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX sessions_token_hash_idx ON public.sessions USING btree (token_hash);


--
-- Name: sessions_user_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX sessions_user_id_idx ON public.sessions USING btree (user_id);


--
-- Name: task_runs_created_at_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX task_runs_created_at_idx ON public.task_runs USING btree (created_at);


--
-- Name: task_runs_outcome_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX task_runs_outcome_idx ON public.task_runs USING btree (outcome);


--
-- Name: task_runs_task_signature_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX task_runs_task_signature_idx ON public.task_runs USING btree (task_signature);


--
-- Name: token_telemetry_created_at_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX token_telemetry_created_at_idx ON public.token_telemetry USING btree (created_at);


--
-- Name: token_telemetry_model_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX token_telemetry_model_idx ON public.token_telemetry USING btree (model);


--
-- Name: token_telemetry_task_type_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX token_telemetry_task_type_idx ON public.token_telemetry USING btree (task_type);


--
-- Name: users_email_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX users_email_idx ON public.users USING btree (email);


--
-- Name: users_username_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX users_username_idx ON public.users USING btree (username);


--
-- Name: mem_edges mem_edges_dst_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_edges
    ADD CONSTRAINT mem_edges_dst_fkey FOREIGN KEY (dst) REFERENCES public.mem_nodes(id) ON DELETE CASCADE;


--
-- Name: mem_edges mem_edges_src_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_edges
    ADD CONSTRAINT mem_edges_src_fkey FOREIGN KEY (src) REFERENCES public.mem_nodes(id) ON DELETE CASCADE;


--
-- Name: mem_trajectory_steps mem_trajectory_steps_trajectory_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_trajectory_steps
    ADD CONSTRAINT mem_trajectory_steps_trajectory_id_fkey FOREIGN KEY (trajectory_id) REFERENCES public.mem_trajectories(id) ON DELETE CASCADE;


--
-- Name: sessions sessions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: mem_canonical; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.mem_canonical ENABLE ROW LEVEL SECURITY;

--
-- Name: mem_canonical mem_canonical_rls; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY mem_canonical_rls ON public.mem_canonical USING ((((scope = ANY (string_to_array(current_setting('app.scopes'::text, true), ','::text))) AND (current_setting('app.scopes'::text, true) <> ''::text)) OR (current_setting('app.bypass_rls'::text, true) = 'on'::text))) WITH CHECK ((((scope = ANY (string_to_array(current_setting('app.scopes'::text, true), ','::text))) AND (current_setting('app.scopes'::text, true) <> ''::text)) OR (current_setting('app.bypass_rls'::text, true) = 'on'::text)));


--
-- PostgreSQL database dump complete
--


