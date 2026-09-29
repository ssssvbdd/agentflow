-- Cascaded routing and hybrid retrieval settings for existing MySQL deployments.
ALTER TABLE agent
    ADD COLUMN enable_semantic_intent_retrieval BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN intent_small_tool_catalog_size INTEGER NOT NULL DEFAULT 8,
    ADD COLUMN intent_deterministic_score_threshold DOUBLE NOT NULL DEFAULT 0.82,
    ADD COLUMN intent_score_margin_threshold DOUBLE NOT NULL DEFAULT 0.20,
    ADD COLUMN intent_candidate_limit INTEGER NOT NULL DEFAULT 12,
    ADD COLUMN intent_semantic_timeout_seconds DOUBLE NOT NULL DEFAULT 3.0;
