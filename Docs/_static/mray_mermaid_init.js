// Simple and reliable Mermaid integration for sphinx-book-theme
(function()
{
    'use strict';

    let mermaidInitialized = false;
    let currentTheme = null;

    // Get current theme from sphinx-book-theme
    function getCurrentTheme()
    {
        const html = document.documentElement;

        // Check for sphinx-book-theme specific attributes/classes
        if (html.getAttribute('data-theme') === 'dark' ||
            html.classList.contains('mode-dark') ||
            document.body.getAttribute('data-theme') === 'dark')
        {
            return 'dark';
        }

        // Check localStorage for sphinx-book-theme preference
        const stored = localStorage.getItem('sphinx-book-theme-mode');
        if (stored === 'dark') return 'dark';

        return 'neutral';
    }

    // Initialize mermaid with theme
    function initMermaid(theme = 'neutral')
    {
        if (!window.mermaid)
        {
            console.warn("Mermaid not loaded yet");
            return false;
        }

        console.log('Initializing Mermaid with theme:', theme);

        try
        {
            mermaid.initialize(
            {
                startOnLoad: false,
                theme: theme,
                securityLevel: 'loose'
            });

            mermaidInitialized = true;
            currentTheme = theme;
            return true;
        }
        catch (error)
        {
            console.error('Mermaid initialization failed:', error);
            return false;
        }
    }

    // Render all mermaid diagrams
    function renderDiagrams()
    {
        if (!mermaidInitialized) return;

        const elements = document.querySelectorAll('.mermaid:not([data-processed])');

        elements.forEach((element, index) =>
        {
            const source = element.textContent.trim();

            // Skip if no content or invalid
            if (!source || source.includes('<svg') || source.includes('data:image'))
            {
                return;
            }

            // Validate it's a proper mermaid diagram
            const validTypes = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram',
                                'stateDiagram', 'erDiagram', 'journey', 'gantt', 'pie',
                                'block-beta', 'timeline'];

            if (!validTypes.some(type => source.includes(type)))
            {
                console.warn('Skipping invalid mermaid diagram:', source.substring(0, 50));
                return;
            }

            const id = `mermaid-diagram-${Date.now()}-${index}`;

            try
            {
                mermaid.render(id, source)
                    .then(result =>
                    {
                        element.innerHTML = result.svg;
                        element.setAttribute('data-processed', 'true');
                        element.setAttribute('data-mermaid-source', source);
                    })
                    .catch(error =>
                    {
                        console.error('Mermaid render error:', error);
                        element.innerHTML = `<div style="color: red; padding: 10px; border: 1px solid red;">
                            Mermaid Error: ${error.message}
                        </div>`;
                    });
            }
            catch (error)
            {
                console.error('Mermaid processing error:', error);
            }
        });
    }

    // Re-render diagrams with new theme
    function reRenderWithTheme(newTheme)
    {
        console.log('Re-rendering diagrams with theme:', newTheme);

        // Reset processed diagrams
        document.querySelectorAll('.mermaid[data-processed]').forEach(element =>
        {
            const source = element.getAttribute('data-mermaid-source');
            if (source)
            {
                element.innerHTML = source;
                element.removeAttribute('data-processed');
            }
        });

        // Re-initialize and render
        if (initMermaid(newTheme))
        {
            setTimeout(renderDiagrams, 100);
        }
    }

    // Watch for theme changes
    function setupThemeWatcher()
    {
        let lastTheme = getCurrentTheme();

        // Watch for attribute changes on html element
        const observer = new MutationObserver(function(mutations)
        {
            mutations.forEach(function(mutation)
            {
                if (mutation.type === 'attributes')
                {
                    const newTheme = getCurrentTheme();
                    if (newTheme !== lastTheme)
                    {
                        lastTheme = newTheme;
                        reRenderWithTheme(newTheme);
                    }
                }
            });
        });

        observer.observe(document.documentElement,
        {
            attributes: true,
            attributeFilter: ['data-theme', 'class']
        });

        observer.observe(document.body,
        {
            attributes: true,
            attributeFilter: ['data-theme', 'class']
        });

        // Listen for storage changes
        window.addEventListener('storage', function(e)
        {
            if (e.key === 'sphinx-book-theme-mode')
            {
                const newTheme = getCurrentTheme();
                if (newTheme !== lastTheme)
                {
                    lastTheme = newTheme;
                    reRenderWithTheme(newTheme);
                }
            }
        });

        // Listen for sphinx-book-theme specific events
        document.addEventListener('sphinx-book-theme-toggle', function()
        {
            setTimeout(() =>
            {
                const newTheme = getCurrentTheme();
                if (newTheme !== lastTheme)
                {
                    lastTheme = newTheme;
                    reRenderWithTheme(newTheme);
                }
            }, 100);
        });
    }

    // Main initialization
    function initialize()
    {
        const theme = getCurrentTheme();

        if (initMermaid(theme))
        {
            renderDiagrams();
            setupThemeWatcher();
        }
        else
        {
            // Retry if mermaid not ready
            setTimeout(initialize, 200);
        }
    }

    // Wait for mermaid to be available
    function waitForMermaid()
    {
        if (window.mermaid)
        {
            initialize();
        }
        else
        {
            setTimeout(waitForMermaid, 100);
        }
    }

    // Start when DOM is ready
    if (document.readyState === 'loading')
    {
        document.addEventListener('DOMContentLoaded', waitForMermaid);
    }
    else
    {
        waitForMermaid();
    }

    // Expose for manual control
    window.mermaidUtils =
    {
        reRender: function()
        {
            reRenderWithTheme(getCurrentTheme());
        },
        getCurrentTheme: getCurrentTheme
    };

})();
