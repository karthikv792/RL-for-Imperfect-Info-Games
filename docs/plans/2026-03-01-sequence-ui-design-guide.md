# Sequence AI — UI & Aesthetic Design Guide

**Date:** 2026-03-01
**Purpose:** Define the visual identity and interaction design for the Sequence board game web app, ensuring tasteful, accessible, and distinctive aesthetics.

---

## Design Philosophy

Sequence is a strategic card game with hidden information. The design should feel:
- **Confident and calm** — like a premium game table, not a casino
- **Spatially clear** — the 10x10 board is the hero; everything else supports it
- **Warmly competitive** — tokens and sequences should feel satisfying to place
- **Intellectually respectful** — this is an AI research showcase, not a toy

> "Good design is as little design as possible." — Dieter Rams

---

## 1. Color System

### Brand Palette

The palette draws from classic card game aesthetics — deep greens (baize/felt), warm ivory, and rich token colors. This is NOT the default purple-gradient-on-white AI aesthetic.

```css
:root {
  /* === BRAND PRIMITIVES === */

  /* Baize Green — the game table */
  --green-900: #0a2618;
  --green-800: #134e33;
  --green-700: #1a6b47;
  --green-600: #22885a;
  --green-500: #2da56e;

  /* Ivory — cards and surfaces */
  --ivory-50:  #fdfcf8;
  --ivory-100: #f9f6ee;
  --ivory-200: #f0ead6;
  --ivory-300: #e5dcc0;

  /* Player 1: Amber/Gold tokens */
  --gold-400: #f59e0b;
  --gold-500: #d97706;
  --gold-600: #b45309;
  --gold-glow: rgba(245, 158, 11, 0.3);

  /* Player 2: Sapphire tokens */
  --sapphire-400: #60a5fa;
  --sapphire-500: #3b82f6;
  --sapphire-600: #2563eb;
  --sapphire-glow: rgba(59, 130, 246, 0.3);
}
```

### Light Theme (Default)

```css
:root, [data-theme="light"] {
  --surface-board: var(--green-800);        /* game board background */
  --surface-page: var(--ivory-50);          /* page background */
  --surface-card: var(--ivory-100);         /* card faces */
  --surface-panel: #ffffff;                 /* side panels */
  --surface-elevated: #ffffff;              /* modals, dropdowns */

  --text-primary: #1a1a2e;
  --text-secondary: #4a5568;
  --text-on-board: var(--ivory-100);        /* text on green board */
  --text-on-gold: #451a03;
  --text-on-sapphire: #1e3a5f;

  --border-card: var(--ivory-300);
  --border-panel: #e2e8f0;

  --player1: var(--gold-500);
  --player1-glow: var(--gold-glow);
  --player2: var(--sapphire-500);
  --player2-glow: var(--sapphire-glow);

  --highlight-legal: rgba(34, 197, 94, 0.25);  /* legal move highlight */
  --highlight-last-move: rgba(251, 191, 36, 0.35);
  --highlight-sequence: rgba(255, 255, 255, 0.6);
}
```

### Dark Theme

```css
[data-theme="dark"] {
  --surface-board: #0a1f14;               /* deeper, richer baize */
  --surface-page: #0f1117;
  --surface-card: #1e2230;                /* dark card faces */
  --surface-panel: #161a24;
  --surface-elevated: #1e2230;

  --text-primary: #e8e8f0;
  --text-secondary: #a0a8b8;
  --text-on-board: #c8d8c8;

  --border-card: #2a3040;
  --border-panel: #252a36;

  --player1: var(--gold-400);             /* slightly lighter for dark bg */
  --player2: var(--sapphire-400);

  --highlight-legal: rgba(34, 197, 94, 0.15);
  --highlight-sequence: rgba(255, 255, 255, 0.2);
}
```

**Key decisions:**
- Gold vs Sapphire for player tokens (not red vs blue — avoids accessibility issues with red-green colorblindness and the aggressive feel of pure red)
- Deep green board — feels like a real game table
- Never pure white or pure black backgrounds

---

## 2. Typography

```css
:root {
  /* Display font — for headings, game title */
  --font-display: 'Space Grotesk', system-ui, sans-serif;

  /* Body font — for UI elements, labels, info */
  --font-body: 'Inter', system-ui, -apple-system, sans-serif;

  /* Mono — for stats, Elo ratings, timers */
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

  /* Card rank/suit text */
  --font-card: 'Playfair Display', Georgia, serif;
}
```

**Type scale for the game:**

| Element | Size | Weight | Font |
|---------|------|--------|------|
| Game title / branding | 28px | 700 | display |
| Section headers (Hand, History) | 14px | 600 | body, uppercase, tracking-wide |
| Card rank on board | 11px | 600 | card (serif) |
| Card suit on board | 10px | 400 | card (serif) |
| Move history entries | 14px | 400 | body |
| Stats / Elo / Timer | 16px | 500 | mono |
| Button labels | 15px | 500 | body |
| Tooltips | 13px | 400 | body |

---

## 3. Board Design

The board is the centerpiece. It should feel like a physical game board — tactile, spatial, warm.

### Board Grid

```
┌─────────────────────────────────────────────────┐
│ ★  2S  3S  4S  5S  6S  7S  8S  9S  ★  │  ★ = corner
│ 6C  5C  4C  3C  2C  AH  KH  QH  10H 10S │     (free cell)
│ 7C  AS  2D  3D  4D  5D  6D  7D  9H  QS  │
│ ...                                       │
│ ★  AD  KD  QD  10D 9D  8D  7D  6D  ★  │
└─────────────────────────────────────────────────┘
```

**Cell design:**
- Each cell is a subtle card-like rectangle with very slight rounded corners (`--radius-sm`: 4px)
- Background: `--surface-card` (ivory in light mode)
- Border: 1px `--border-card` — subtle separation, not harsh grid lines
- Card text: rank top-left, suit symbol bottom-right (like a real playing card, miniaturized)
- Cell gap: 2px — tight but visible separation

**Board background:**
- Deep green (`--surface-board`) with subtle radial gradient lighter toward center
- Optional: very faint felt texture via CSS background-image (subtle noise pattern at 3-5% opacity)

**Corners:**
- Distinct visual: diagonal pattern or small star icon
- No card text — they're clearly "free" cells
- Subtle shimmer or different background to indicate they count for any player

### Tokens

Tokens are the most important visual element — they represent the core gameplay.

```
Token specifications:
- Shape: circle, sized to ~70% of cell width
- Player 1 (Gold): solid fill, subtle inner shadow for depth
- Player 2 (Sapphire): solid fill, subtle inner shadow for depth
- Border: 2px slightly darker variant of token color
- Placement animation: scale from 0 → 1.1 → 1.0 (spring ease, 250ms)
- Removal animation: scale to 0.8, fade out (150ms)
- Glow: when part of a completed sequence, subtle outer glow (box-shadow)
```

```css
.token {
  width: 70%;
  height: 70%;
  border-radius: var(--radius-full);
  position: absolute;
  top: 15%;
  left: 15%;
  transition: transform var(--duration-base) var(--ease-spring),
              opacity var(--duration-fast) var(--ease-out);
}

.token--player1 {
  background: radial-gradient(circle at 35% 35%, #fbbf24, var(--player1));
  border: 2px solid var(--gold-600);
  box-shadow: inset 0 2px 4px rgba(255,255,255,0.3),
              inset 0 -2px 4px rgba(0,0,0,0.15);
}

.token--player2 {
  background: radial-gradient(circle at 35% 35%, #93c5fd, var(--player2));
  border: 2px solid var(--sapphire-600);
  box-shadow: inset 0 2px 4px rgba(255,255,255,0.3),
              inset 0 -2px 4px rgba(0,0,0,0.15);
}

.token--in-sequence {
  box-shadow: 0 0 12px var(--player1-glow);  /* or player2-glow */
  animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes token-place {
  0%   { transform: scale(0); opacity: 0; }
  70%  { transform: scale(1.1); opacity: 1; }
  100% { transform: scale(1); }
}

@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 8px var(--player1-glow); }
  50%      { box-shadow: 0 0 16px var(--player1-glow); }
}
```

### Legal Move Highlights

When it's the human's turn, highlight legal moves:

- **Playable cells**: Subtle green overlay (`--highlight-legal`) — NOT a harsh border
- **Hovered legal cell**: Brighter highlight + slight scale-up of cell (1.02x)
- **Selected card → matching positions**: Matching cells get a stronger highlight pulse
- **Last AI move**: Brief flash of `--highlight-last-move`, then settle to a subtle indicator dot

### Completed Sequences

When 5-in-a-row is achieved:
- The 5 tokens get the `pulse-glow` animation briefly (2 seconds)
- A subtle connecting line (SVG path) draws between the 5 cells
- The line uses the player's color at 50% opacity
- Tokens in a completed sequence get a permanent subtle glow

---

## 4. Layout

### Page Structure

```
┌────────────────────────────────────────────────────────┐
│  SEQUENCE AI                    [Agent: ViT ▼] [⚙]   │  ← Top bar
├────────────────┬──────────────────────┬────────────────┤
│                │                      │                │
│   Your Hand    │                      │   Game Info    │
│   [cards...]   │      10x10 BOARD     │   Turn: You    │
│                │                      │   Sequences:   │
│   [5 cards     │    (the hero —       │   🟡 1  🔵 0   │
│    laid out    │     dominates the    │                │
│    face up]    │     visual space)    │   AI Thinking  │
│                │                      │   ████░░ 1.2s  │
│                │                      │                │
│                │                      │   Move History │
│                │                      │   1. 🟡 2S→(0,1)│
│                │                      │   2. 🔵 J2→(3,5)│
│                │                      │   ...          │
│                │                      │                │
├────────────────┴──────────────────────┴────────────────┤
│  [New Game]                          [Elo: 1523]      │  ← Bottom bar
└────────────────────────────────────────────────────────┘
```

**Key layout principles:**
- Board takes **60-65% of viewport width** — it's the hero (Pillar 1: Hierarchy & Focus)
- Left panel: player's hand (cards fanned out or in a row)
- Right panel: game info, move history
- Responsive: on mobile, panels collapse below the board
- Board always visible without scrolling on desktop

### Card Hand Display

Cards in the player's hand should feel tactile:

```
Card design (in hand):
- Slightly larger than board cells (80x112px at default zoom)
- Ivory background with subtle card texture
- Rank in top-left and bottom-right (like real playing cards)
- Suit symbol colored (♠ black, ♥ red, ♦ red, ♣ black)
- Hover: lift up 8px + slight shadow increase (translate + shadow transition)
- Selected: lifted 16px, golden border ring
- Unplayable cards: slightly desaturated (opacity 0.6)
- Serif font (Playfair Display) for rank/suit — classic card feel
```

```css
.hand-card {
  width: 80px;
  height: 112px;
  background: var(--surface-card);
  border: 1px solid var(--border-card);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
  transition: transform var(--duration-fast) var(--ease-spring),
              box-shadow var(--duration-fast) var(--ease-out);
  cursor: pointer;
  position: relative;
}

.hand-card:hover {
  transform: translateY(-8px);
  box-shadow: var(--shadow-lg);
}

.hand-card--selected {
  transform: translateY(-16px);
  box-shadow: 0 0 0 2px var(--player1), var(--shadow-lg);
}

.hand-card--unplayable {
  opacity: 0.5;
  cursor: not-allowed;
  filter: grayscale(0.3);
}
```

---

## 5. Interaction Design

### Move Flow (Human Player)

```
1. It's your turn → subtle pulse on your hand panel
2. Hover a card → card lifts, matching board positions highlight
3. Click a card → card stays lifted, board highlights solidify
4. Hover a highlighted board cell → cell scales up slightly
5. Click a board cell → token places with spring animation
6. Card slides out of hand → new card slides in from deck
7. Turn indicator switches → AI thinking indicator appears
```

**Timing budget:**
- Card hover lift: 150ms ease-spring
- Token placement: 250ms ease-spring (scale 0→1.1→1.0)
- Card deal-in: 300ms ease-out (slide from right)
- Turn switch: 200ms crossfade

### AI Thinking State

When the AI is computing:

```
- Thin progress bar below the board (indeterminate shimmer, not fake percentage)
- Text: "ViT is thinking..." (use agent name)
- Optional: subtle heatmap overlay on the board showing the AI's policy distribution
  (very low opacity, purely informational — like watching it think)
- AI move lands with the same token animation but preceded by a brief
  "consideration" highlight where 2-3 candidate cells flash subtly before
  the chosen one gets the token
```

### Spectator Mode

When watching AI vs AI:

```
- Both hands hidden (no information to show)
- Larger board (no hand panel needed)
- Auto-play at configurable speed (0.5s, 1s, 2s per move)
- Play/Pause controls
- Optional policy heatmap overlay toggle
- Move counter and turn indicator more prominent
```

---

## 6. Motion Design

All motion serves a purpose. No decorative animation.

| Action | Animation | Duration | Easing |
|--------|-----------|----------|--------|
| Token place | Scale 0→1.1→1.0 | 250ms | spring |
| Token remove | Scale 1→0.8, fade out | 150ms | ease-in |
| Card hover (hand) | TranslateY -8px | 150ms | spring |
| Card select | TranslateY -16px + border | 150ms | spring |
| Card deal-in | TranslateX 100%→0 + fade | 300ms | ease-out |
| Card play-out | Scale→0.8, translateY +20px, fade | 200ms | ease-in |
| Legal move highlight | Opacity 0→1 | 200ms | ease-out |
| Sequence complete | Glow pulse (2 cycles) | 2000ms | ease-in-out |
| Page transition | Fade 0→1 | 300ms | ease-out |
| AI thinking bar | Shimmer loop | continuous | linear |

```css
/* Respect user preference */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## 7. Game Over State

When someone wins:

```
1. Winning sequence tokens pulse-glow (2 cycles)
2. Connecting line animates between the 5 tokens (SVG stroke-dasharray)
3. Subtle confetti burst (small, tasteful — 20-30 particles, player's color only)
4. Overlay with result: "Gold Wins!" or "Sapphire Wins!" or "You Win!/You Lost"
5. Stats shown: moves played, time, sequences formed
6. CTA: [Play Again] [View Replay] [Change Agent]
```

The overlay should NOT be a full-screen takeover. It should be a centered card (modal-like) that still shows the board behind it, slightly dimmed.

---

## 8. Responsive Behavior

| Breakpoint | Layout |
|------------|--------|
| > 1280px | 3-column: Hand | Board | Info |
| 768-1280px | Board top, Hand + Info in tabs below |
| < 768px | Board fills width, Hand as horizontal scroll below, Info in collapsible drawer |

On mobile:
- Cards in hand: horizontal scroll, slightly overlapping (like a card fan)
- Board: pinch-to-zoom enabled
- Tap a card → board highlights appear
- Tap a highlighted cell → place token
- No hover states (touch device)

---

## 9. Accessibility

Non-negotiable requirements:

- All token colors have **shape distinction** too (Gold = circle, Sapphire = circle with inner dot, or different border style) for colorblind users
- Board cells have aria-labels: "Row 3, Column 5: 4 of Hearts, occupied by Gold"
- Keyboard navigation: arrow keys move between board cells, Enter to place
- Screen reader announces: "Gold placed token at Row 3, Column 5" and "Sapphire is thinking..."
- Minimum 4.5:1 contrast ratio on all text
- Focus indicators clearly visible (2px accent outline, 2px offset)
- High contrast mode supported via `prefers-contrast: more`

---

## 10. Empty / Zero States

### New Game (no moves yet)

```
Board: All cells visible, no tokens
Hand: 5 cards dealt with a stagger animation (each card slides in 100ms apart)
Info panel: "Make your move — select a card, then tap a matching position"
```

### First Visit (no account)

```
Hero section: "Challenge the AI at Sequence"
Brief rules summary (collapsible)
CTA: [Start Playing] — no sign-up required
Agent picker: visual cards for each AI type with brief description
```

---

## 11. Design Token Reference

The full token CSS file should be placed at `webapp/src/styles/tokens.css` and imported globally. It includes:

- Spacing scale (8px base)
- Typography scale
- Color system (light + dark themes)
- Border radius scale
- Shadow system
- Motion/easing tokens
- Z-index stacking system

All values in the app MUST reference tokens. No magic numbers.

---

## Anti-Pattern Checklist

Before shipping, verify:

```
□ No purple-to-blue gradient (we use green baize + ivory)
□ No identical border-radius on everything (cards: md, tokens: full, board: lg)
□ No shadow on every element (only cards in hand and elevated panels)
□ No placeholder-as-label in any forms
□ No auto-playing animations (all motion is response to user action)
□ Component library defaults customized (not stock shadcn/MUI)
□ Empty states designed (not blank)
□ Mobile is designed, not reflowed
□ All magic numbers replaced with tokens
□ Semantic HTML throughout
□ Tested in actual dark environment
□ Tested with screen reader
□ Keyboard navigable end-to-end
```
