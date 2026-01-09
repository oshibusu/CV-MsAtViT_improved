module.exports = {
	stylesheet: [
		'https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-light.min.css',
		'https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css'
	],
	body_class: 'markdown-body',
	pdf_options: {
		format: 'A4',
		margin: '20mm',
		printBackground: true
	},
	script: [
		{ url: 'https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js' },
		{ url: 'https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js' },
		{ url: 'https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js' },
		{
			content: `
			document.addEventListener("DOMContentLoaded", function() {
				// Render Math
				renderMathInElement(document.body, {
					delimiters: [
						{left: "$$", right: "$$", display: true},
						{left: "$", right: "$", display: false}
					]
				});

				// Render Mermaid
				// 1. Convert <pre><code class="mermaid"> (or hljs mermaid) to <div class="mermaid">
				const mermaidBlocks = document.querySelectorAll('code.mermaid');
				mermaidBlocks.forEach(block => {
					const pre = block.parentElement;
					const div = document.createElement('div');
					div.className = 'mermaid';
					// Decode HTML entities if needed, though textContent usually handles it
					div.textContent = block.textContent;
					div.style.display = 'flex';
					div.style.justifyContent = 'center';
					pre.replaceWith(div);
				});

				// 2. Initialize Mermaid
				mermaid.initialize({ startOnLoad: true });
			});
		` }
	]
};
