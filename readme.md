flowchart TD
%% ===== Encoder Path =====
subgraph Encoder["Encoding Module"]
    style Encoder fill:#e0ffe0,stroke:#33aa33,stroke-width:2px

    H[Channel Matrix H]
    SE[StateEncoder &#40;CNN&#41;]
    W[Previous Beamformer W_&#123;t-1&#125;]
    AE[ActionEncoder &#40;Linear&#41;]
    CAT[Concat h and w]
    TP[Token Projector]
    PE[Add Positional Embedding]
    TF[Transformer Blocks × N]
    LN[Layer Normalization]
    DEC[Action Decoder &#40;Linear&#41;]
    W_next[Predicted Beamformer W_t]

    H --> SE
    W --> AE
    SE --> CAT
    AE --> CAT
    CAT --> TP
    TP --> PE
    PE --> TF
    TF --> LN
    LN --> DEC
    DEC --> W_next
end

%% ===== Training Loop =====
subgraph TrainingLoop["Training Loop"]
    style TrainingLoop fill:#e0f0ff,stroke:#3399cc,stroke-width:2px

    A[Sample batch of H]
    B[Random Initialize W_0]
    C[Loop over t = 1 to T]
    D[Build token sequence: h, W_0 to W_&#123;t-1&#125;]
    E[Predict W_t using Transformer]
    F[Compute sum rate R&#40;H, W_t&#41;]
    G{t &lt; T?}
    H[Compute total sum rate loss]
    I[Backpropagation and update Transformer weights]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G -- Yes --> C
    G -- No --> H
    H --> I
end

%% ===== Connections between blocks =====
D --> CAT
E --> W_next
