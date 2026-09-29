-- Structured intent routing configuration for existing MySQL deployments.
ALTER TABLE agent
    ADD COLUMN enable_intent_router BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN intent_confidence_threshold DOUBLE NOT NULL DEFAULT 0.65,
    ADD COLUMN intent_history_messages INTEGER NOT NULL DEFAULT 12;
