```mermaid
  graph TD;
      A[Client] --> B[Load Balancer];
      B --> C[Web Server];
      C --> D[Application Server];
      D --> E[Database];
      C --> F[Cache];
```