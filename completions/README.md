# MechanicsDSL Shell Completions

Shell completion scripts for the `mechanicsdsl` CLI.

## Installation

### Bash

Add to your `~/.bashrc`:

```bash
eval "$(_MECHANICSDSL_COMPLETE=bash_source mechanicsdsl)"
```

Or save to a file:

```bash
_MECHANICSDSL_COMPLETE=bash_source mechanicsdsl > ~/.mechanicsdsl-complete.bash
echo '. ~/.mechanicsdsl-complete.bash' >> ~/.bashrc
```

### Zsh

Add to your `~/.zshrc`:

```zsh
eval "$(_MECHANICSDSL_COMPLETE=zsh_source mechanicsdsl)"
```

Or save to a file:

```zsh
_MECHANICSDSL_COMPLETE=zsh_source mechanicsdsl > ~/.mechanicsdsl-complete.zsh
echo '. ~/.mechanicsdsl-complete.zsh' >> ~/.zshrc
```

### Fish

Save to Fish completions directory:

```fish
_MECHANICSDSL_COMPLETE=fish_source mechanicsdsl > ~/.config/fish/completions/mechanicsdsl.fish
```

---

## Manual Completion Scripts

If the automatic generation doesn't work, use these manual scripts.

### Bash (Manual)

Save as `~/.mechanicsdsl-complete.bash`:

```bash
_mechanicsdsl_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Commands
    if [[ ${COMP_CWORD} == 1 ]]; then
        opts="compile run export validate info --help --version"
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
    
    # Subcommand options
    case "${COMP_WORDS[1]}" in
        compile)
            case "${prev}" in
                --target|-t)
                    opts="cpp cuda rust julia fortran matlab javascript wasm python arduino openmp"
                    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
                    return 0
                    ;;
                --output|-o)
                    COMPREPLY=( $(compgen -d -- ${cur}) )
                    return 0
                    ;;
                *)
                    if [[ ${cur} == -* ]]; then
                        opts="--target --output --help"
                        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
                    else
                        COMPREPLY=( $(compgen -f -X '!*.mdsl' -- ${cur}) )
                    fi
                    return 0
                    ;;
            esac
            ;;
        run)
            case "${prev}" in
                --t-span)
                    COMPREPLY=( $(compgen -W "0,10 0,20 0,30 0,100" -- ${cur}) )
                    return 0
                    ;;
                --points|-n)
                    COMPREPLY=( $(compgen -W "100 500 1000 5000 10000" -- ${cur}) )
                    return 0
                    ;;
                --output|-o)
                    COMPREPLY=( $(compgen -f -X '!*.json' -- ${cur}) )
                    return 0
                    ;;
                *)
                    if [[ ${cur} == -* ]]; then
                        opts="--t-span --points --animate --output --help"
                        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
                    else
                        COMPREPLY=( $(compgen -f -X '!*.mdsl' -- ${cur}) )
                    fi
                    return 0
                    ;;
            esac
            ;;
        export)
            case "${prev}" in
                --format|-f)
                    opts="json csv numpy"
                    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
                    return 0
                    ;;
                --output|-o)
                    COMPREPLY=( $(compgen -f -- ${cur}) )
                    return 0
                    ;;
                *)
                    if [[ ${cur} == -* ]]; then
                        opts="--format --output --t-span --points --help"
                        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
                    else
                        COMPREPLY=( $(compgen -f -X '!*.mdsl' -- ${cur}) )
                    fi
                    return 0
                    ;;
            esac
            ;;
        validate)
            if [[ ${cur} == -* ]]; then
                opts="--help"
                COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            else
                COMPREPLY=( $(compgen -f -X '!*.mdsl' -- ${cur}) )
            fi
            return 0
            ;;
        info)
            if [[ ${cur} == -* ]]; then
                opts="--help"
                COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            fi
            return 0
            ;;
    esac
}

complete -F _mechanicsdsl_completion mechanicsdsl
```

### Zsh (Manual)

Save as `~/.mechanicsdsl-complete.zsh`:

```zsh
#compdef mechanicsdsl

_mechanicsdsl() {
    local -a commands
    commands=(
        'compile:Compile DSL to target language'
        'run:Run a simulation'
        'export:Export simulation results'
        'validate:Validate a DSL file'
        'info:Show version and system info'
    )
    
    local -a targets
    targets=(cpp cuda rust julia fortran matlab javascript wasm python arduino openmp)
    
    local -a formats
    formats=(json csv numpy)
    
    _arguments -C \
        '1: :->command' \
        '*:: :->args'
    
    case $state in
        command)
            _describe -t commands 'mechanicsdsl command' commands
            ;;
        args)
            case $words[1] in
                compile)
                    _arguments \
                        '1:input file:_files -g "*.mdsl"' \
                        '--target[Target language]:target:($targets)' \
                        '-t[Target language]:target:($targets)' \
                        '--output[Output directory]:directory:_files -/' \
                        '-o[Output directory]:directory:_files -/'
                    ;;
                run)
                    _arguments \
                        '1:input file:_files -g "*.mdsl"' \
                        '--t-span[Time span]:span:' \
                        '--points[Number of points]:number:' \
                        '-n[Number of points]:number:' \
                        '--animate[Show animation]' \
                        '-a[Show animation]' \
                        '--output[Output file]:file:_files -g "*.json"' \
                        '-o[Output file]:file:_files -g "*.json"'
                    ;;
                export)
                    _arguments \
                        '1:input file:_files -g "*.mdsl"' \
                        '--format[Output format]:format:($formats)' \
                        '-f[Output format]:format:($formats)' \
                        '--output[Output file]:file:_files' \
                        '-o[Output file]:file:_files' \
                        '--t-span[Time span]:span:' \
                        '--points[Number of points]:number:' \
                        '-n[Number of points]:number:'
                    ;;
                validate)
                    _arguments \
                        '1:input file:_files -g "*.mdsl"'
                    ;;
            esac
            ;;
    esac
}

_mechanicsdsl "$@"
```

### Fish (Manual)

Save as `~/.config/fish/completions/mechanicsdsl.fish`:

```fish
# mechanicsdsl completions for Fish shell

# Subcommands
complete -c mechanicsdsl -n "__fish_use_subcommand" -a compile -d "Compile DSL to target language"
complete -c mechanicsdsl -n "__fish_use_subcommand" -a run -d "Run a simulation"
complete -c mechanicsdsl -n "__fish_use_subcommand" -a export -d "Export simulation results"
complete -c mechanicsdsl -n "__fish_use_subcommand" -a validate -d "Validate a DSL file"
complete -c mechanicsdsl -n "__fish_use_subcommand" -a info -d "Show version and system info"

# compile subcommand
complete -c mechanicsdsl -n "__fish_seen_subcommand_from compile" -s t -l target -d "Target language" -xa "cpp cuda rust julia fortran matlab javascript wasm python arduino openmp"
complete -c mechanicsdsl -n "__fish_seen_subcommand_from compile" -s o -l output -d "Output directory" -xa "(__fish_complete_directories)"

# run subcommand
complete -c mechanicsdsl -n "__fish_seen_subcommand_from run" -l t-span -d "Time span (start,end)"
complete -c mechanicsdsl -n "__fish_seen_subcommand_from run" -s n -l points -d "Number of points"
complete -c mechanicsdsl -n "__fish_seen_subcommand_from run" -s a -l animate -d "Show animation"
complete -c mechanicsdsl -n "__fish_seen_subcommand_from run" -s o -l output -d "Output JSON file"

# export subcommand
complete -c mechanicsdsl -n "__fish_seen_subcommand_from export" -s f -l format -d "Output format" -xa "json csv numpy"
complete -c mechanicsdsl -n "__fish_seen_subcommand_from export" -s o -l output -d "Output file"
complete -c mechanicsdsl -n "__fish_seen_subcommand_from export" -l t-span -d "Time span"
complete -c mechanicsdsl -n "__fish_seen_subcommand_from export" -s n -l points -d "Number of points"

# File completions for .mdsl files
complete -c mechanicsdsl -n "__fish_seen_subcommand_from compile run export validate" -a "(find . -name '*.mdsl' 2>/dev/null)"
```
