import WebSocket from 'ws';
import {ConnectorSupervisor} from './connector-supervisor.ts';

/** Node companion adapter. The installer/local daemon owns credential loading;
 * the browser never receives these headers. Node runtime packaging is a
 * separate installer gate, not an end-user npm setup requirement.
 */
export function createNodeConnector(options) {
  return new ConnectorSupervisor({...options,dial:config=>new WebSocket(config.url,config.options)});
}
