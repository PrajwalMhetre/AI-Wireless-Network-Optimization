-- Metadata-only PostgreSQL schema. Large CSI tensors belong in object storage.
CREATE TABLE IF NOT EXISTS experiments (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS scenarios (
    id UUID PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(id),
    environment TEXT NOT NULL,
    config JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY,
    scenario_id UUID REFERENCES scenarios(id),
    model_version TEXT NOT NULL,
    phases JSONB NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    ood_score DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS optimization_runs (
    id UUID PRIMARY KEY,
    prediction_id UUID REFERENCES predictions(id),
    decision TEXT NOT NULL,
    optimizer_called BOOLEAN NOT NULL,
    metadata JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY,
    optimization_run_id UUID REFERENCES optimization_runs(id),
    values JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS uncertainty_results (
    id UUID PRIMARY KEY,
    prediction_id UUID REFERENCES predictions(id),
    uncertainty DOUBLE PRECISION NOT NULL,
    calibration JSONB
);

CREATE TABLE IF NOT EXISTS active_learning_samples (
    id UUID PRIMARY KEY,
    scenario_id UUID REFERENCES scenarios(id),
    reason TEXT NOT NULL,
    improvement DOUBLE PRECISION,
    artifact_uri TEXT
);

CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY,
    version TEXT NOT NULL,
    artifact_uri TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
