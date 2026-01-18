# JSON Schemas

Formal schemas for MechanicsDSL configuration and data files.

## Available Schemas

| Schema | Description |
|--------|-------------|
| `config.schema.json` | Project configuration file validation |
| `simulation-output.schema.json` | Simulation result file format |

## Usage

### VS Code

Add to your `.vscode/settings.json`:

```json
{
  "json.schemas": [
    {
      "fileMatch": ["mechanicsdsl.json", ".mechanicsdslrc"],
      "url": "./schemas/config.schema.json"
    }
  ]
}
```

### Python Validation

```python
import json
import jsonschema

with open('schemas/config.schema.json') as f:
    schema = json.load(f)

with open('myproject/mechanicsdsl.json') as f:
    config = json.load(f)

jsonschema.validate(config, schema)
```

### CLI Validation

```bash
pip install jsonschema
jsonschema -i myconfig.json schemas/config.schema.json
```
