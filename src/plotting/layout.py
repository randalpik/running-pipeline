"""Default Plotly layout helpers shared by every plot in src/plots/.

Each plot script still calls ``fig.update_layout(...)`` for its own title,
axes, legend — this just stamps the dark-theme defaults so the same
paper_bgcolor/plot_bgcolor/font boilerplate isn't copy-pasted six times.
"""
from __future__ import annotations

from .tokens import BG, FG, FG_SOFT


# Vertical room reserved at the top of every plot for the title block.
# Pairs with title_block(...) below. Bumping this without bumping the title
# y-anchor will leave more whitespace above the title; keep them aligned.
TITLE_MARGIN_TOP = 90


def apply_default_layout(fig, **overrides):
    """Stamp dark-theme defaults onto ``fig`` then apply caller overrides.

    ``overrides`` is forwarded to ``fig.update_layout(...)`` after defaults
    are set, so callers can override anything they want using the same
    kwarg shape as ``update_layout``. Pass each plot's title/axes/legend/
    margin overrides exactly as you would to ``update_layout``.
    """
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=FG),
        autosize=True,
    )
    if overrides:
        fig.update_layout(**overrides)
    return fig


def title_block(main: str, subtitle: str | None = None) -> dict:
    """Build a unified title dict for ``fig.update_layout(title=...)``.

    Plain (non-bold) main title at 18px so it reads visibly distinct from
    in-plot text (axis titles ~12px, subtitle 13px, FG_SOFT). Optional
    subtitle on a second line. Anchored top-left to align with the
    tab-shell brand at the same corner of the viewport. The caller is
    responsible for setting ``margin=dict(t=TITLE_MARGIN_TOP, ...)`` so
    the title has enough vertical room.

    Subtitles can include ``<i>``, ``·``, etc. — content is passed through.
    """
    if subtitle:
        text = (f'{main}'
                f'<br><sub style="font-size:13px;color:{FG_SOFT}">'
                f'{subtitle}</sub>')
    else:
        text = main
    return dict(
        text=text,
        x=0.01, xanchor='left',
        y=0.99, yanchor='top',
        font=dict(size=18),
    )
