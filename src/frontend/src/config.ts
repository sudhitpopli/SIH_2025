/**
 * [MECHANISM: DYNAMIC BACKEND ROUTING]
 * This config allows the frontend to point to different backends (Local vs Render) 
 * without changing the code.
 * 
 * Vite requires 'VITE_' prefix for environment variables to be visible in client code.
 */

const backendHost = import.meta.env.VITE_BACKEND_URL || 'localhost:8000';

// Detect if we should use secure protocols (https/wss)
// Usually true if the host doesn't contain 'localhost'
const isSecure = !backendHost.includes('localhost');

export const API_BASE_URL = `${isSecure ? 'https' : 'http'}://${backendHost}`;
export const WS_BASE_URL = `${isSecure ? 'wss' : 'ws'}://${backendHost}`;

console.log(`[CONFIG] Backend initialized at ${API_BASE_URL}`);
