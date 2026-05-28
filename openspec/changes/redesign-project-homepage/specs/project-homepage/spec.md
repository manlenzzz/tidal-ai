# Project Homepage Spec

## ADDED Requirements

### Requirement: Toolkit-first homepage
The homepage SHALL present TIDAL as an open-source efficient large-model systems toolkit rather than a personal paper archive.

#### Scenario: First viewport communicates use
- **WHEN** a visitor opens the homepage
- **THEN** they see the project name, concise positioning, workflow categories, and a practical code example.

### Requirement: Scan-friendly workflow entry points
The homepage SHALL expose public workflow entry points for compression, adaptation, serving, and future post-training scope.

#### Scenario: User scans available capabilities
- **WHEN** a visitor reaches the workflow section
- **THEN** they can identify CAP/QPruner compression, RankAdaptor adaptation, LoRA serving integrations, and upcoming post-training/RL scope.

### Requirement: Visible citation surface
The homepage SHALL keep toolkit and method citations visible without overwhelming the first viewport.

#### Scenario: User needs citation information
- **WHEN** a visitor navigates to Citation
- **THEN** toolkit citation and all six method-paper citation blocks are available.

### Requirement: Responsive static site
The homepage SHALL render cleanly on desktop and mobile with no incoherent overlap.

#### Scenario: Mobile visitor opens the page
- **WHEN** viewport width is small
- **THEN** navigation, hero, cards, method matrix, and citations remain readable in a single-column layout.
