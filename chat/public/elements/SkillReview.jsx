// One table for skill review. Each row runs its action in place through the
// Python action callbacks (keep_new / merge / revert) and shows the result.
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";

export default function SkillReview() {
  const pending = props.decision === "pending";
  const [targets, setTargets] = useState({});
  const [status, setStatus] = useState({});
  const [busy, setBusy] = useState(null);

  async function run(action, name, extra = {}) {
    setBusy(name);
    try {
      const result = await callAction({ name: action, payload: { name, ...extra } });
      const answer = (result && result.response) || {};
      setStatus((s) => ({ ...s, [name]: { ok: !!answer.ok, text: answer.message || "?" } }));
    } catch (error) {
      setStatus((s) => ({ ...s, [name]: { ok: false, text: String(error) } }));
    } finally {
      setBusy(null);
    }
  }

  function merge(name) {
    const canonical = (targets[name] || "").trim();
    if (canonical) run("merge", name, { canonical });
  }

  return (
    <div className="w-full overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Skill · lý do của LLM</TableHead>
            {pending ? <TableHead>Conf.</TableHead> : <TableHead>Đã gộp vào</TableHead>}
            <TableHead className="whitespace-nowrap">Thao tác</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {props.items.map((item) => {
            const done = status[item.name];
            return (
              <TableRow key={item.name}>
                <TableCell>
                  <div className="font-medium">{item.name}</div>
                  <div className="text-muted-foreground text-xs">{item.reason || ""}</div>
                </TableCell>
                <TableCell>
                  {pending
                    ? (item.confidence == null ? "—" : item.confidence.toFixed(2))
                    : (item.skill || "—")}
                </TableCell>
                <TableCell>
                  {done && done.ok ? (
                    <Badge variant="secondary">✅ {done.text}</Badge>
                  ) : pending ? (
                    <div className="flex gap-2 items-center whitespace-nowrap">
                      <Input
                        className="h-8 w-36"
                        placeholder="gộp vào…"
                        value={targets[item.name] || ""}
                        onChange={(e) => setTargets((t) => ({ ...t, [item.name]: e.target.value }))}
                        onKeyDown={(e) => e.key === "Enter" && merge(item.name)}
                      />
                      <Button size="sm" disabled={busy === item.name || !(targets[item.name] || "").trim()}
                        onClick={() => merge(item.name)}>Gộp</Button>
                      <Button size="sm" variant="outline" disabled={busy === item.name}
                        onClick={() => run("keep_new", item.name)}>Giữ riêng</Button>
                    </div>
                  ) : (
                    <Button size="sm" variant="outline" disabled={busy === item.name}
                      onClick={() => run("revert", item.name)}>Hoàn tác</Button>
                  )}
                  {done && !done.ok ? (
                    <div className="text-destructive text-sm mt-1">{done.text}</div>
                  ) : null}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
