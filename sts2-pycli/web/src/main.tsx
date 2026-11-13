import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import { registerWorkbenchServiceWorker } from './pwa'
import { installArtPlaceholder } from './artPlaceholder'
import { currentAssetsBase } from './api'
import './phone.css'

registerWorkbenchServiceWorker()
// A checkout without extracted game art (see docs/bring-your-own-game.md)
// renders a neutral tile wherever an image 404s instead of a broken frame.
installArtPlaceholder(document, currentAssetsBase)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
