def main [splats: string] {
  let splats = open $splats
  let ratio: float = $splats.aspect_ratio
  let height = 900.
  let width = 900. * $ratio

  let controls = $splats
  | get splats
  | each {
    let x = $in.pos.0 * $width - $width / 2.
    let y = $in.pos.1 * $height - $height / 2.
    let w = $in.size.0 * $width
    let h = $in.size.1 * $height
    let rot = $in.rot * (180. / 3.1415926535897932384626)
    let r = $in.color.0 / $in.color.3 * 255. | into int
    let g = $in.color.1 / $in.color.3 * 255. | into int
    let b = $in.color.2 / $in.color.3 * 255. | into int
    let a = $in.color.3 * 255. | into int
    let sx = $in.scale.0
    let sy = $in.scale.1

    $"{ ($x), ($y), ($w), ($h), ($rot), ($r), ($g), ($b), ($a), ($sx), ($sy) }"
  }
  | str join ",\n    ";

  let r = $splats.background.0 / $splats.background.3 * 255. | into int
  let g = $splats.background.1 / $splats.background.3 * 255. | into int
  let b = $splats.background.2 / $splats.background.3 * 255. | into int
  let a = $splats.background.3 * 255. | into int
  let background = $"{ ($r), ($g), ($b), ($a) }";
  
$"return {
  aspect_ratio = ($splats.aspect_ratio),
  background = ($background),
  splats = {
    ($controls),
  },
}"
  | save splats.lua -f
}
