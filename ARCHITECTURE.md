```mermaid
graph LR
    A[User] -->|Requests| B[Server]
    B --> C[Database]
    B --> D[Cache]
    C -->|Reads/Writes| E[Data Store]
    D -->|Fetches| E
```