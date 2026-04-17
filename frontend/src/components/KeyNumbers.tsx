export function KeyNumbers({ numbers }: { numbers: string[] }) {
  if (!numbers.length) return null;
  return (
    <ul className="keynumbers">
      {numbers.map((n, i) => <li key={i} className="chip chip-readonly">{n}</li>)}
    </ul>
  );
}
