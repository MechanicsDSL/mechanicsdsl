# MechanicsDSL VS Code Extension

Syntax highlighting for the MechanicsDSL physics domain-specific language.

## Features

- Syntax highlighting for `.mdsl` and `.mechanics` files
- Highlights DSL commands (`\system`, `\lagrangian`, `\hamiltonian`, etc.)
- Greek letters support (`\theta`, `\omega`, etc.)
- Mathematical functions (`\frac`, `\sqrt`, `\sin`, etc.)
- Comment support (lines starting with `%`)

## Installation

### From VSIX (Local Installation)

1. Package the extension:
   ```bash
   cd vscode-extension
   npm install -g vsce
   vsce package
   ```

2. Install in VS Code:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Click "..." menu → "Install from VSIX..."
   - Select the generated `.vsix` file

### Manual Installation

Copy the `vscode-extension` folder to:
- Windows: `%USERPROFILE%\.vscode\extensions\mechanicsdsl-0.1.0`
- macOS/Linux: `~/.vscode/extensions/mechanicsdsl-0.1.0`

## Usage

Create a file with `.mdsl` extension and start writing MechanicsDSL code:

```latex
\system{pendulum}
\defvar{theta}{Angle}{rad}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

% Lagrangian for simple pendulum
\lagrangian{\frac{1}{2} m l^2 \dot{\theta}^2 + m g l \cos(\theta)}

\initial{theta=0.5, theta_dot=0}
```

## License

MIT License - see main repository for details.
