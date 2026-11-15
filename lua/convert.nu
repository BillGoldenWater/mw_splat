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

    mut r = $in.color.0
    mut g = $in.color.1
    mut b = $in.color.2
    let a = $in.color.3
    if $in.color.3 != 0. {
      $r /= $a
      $g /= $a
      $b /= $a
    } else {
      $r = 0.;
      $g = 0.;
      $b = 0.;
    }
    let r = $r * 255. | into int
    let g = $g * 255. | into int
    let b = $b * 255. | into int
    let a = $a * 255. | into int

    let sx = $in.scale.0
    let sy = $in.scale.1

    $"{ ($x), ($y), ($w), ($h), ($rot), ($r), ($g), ($b), ($a), ($sx), ($sy) }"
  }
  | str join ",\n    ";

  let background = if $splats.background.3 != 0. {
    let r = $splats.background.0 / $splats.background.3 * 255. | into int
    let g = $splats.background.1 / $splats.background.3 * 255. | into int
    let b = $splats.background.2 / $splats.background.3 * 255. | into int
    let a = $splats.background.3 * 255. | into int
    $"{ ($r), ($g), ($b), ($a) }"
  } else {
    $"{ 0, 0, 0, 0 }"
  }
  
$"return {
  aspect_ratio = ($splats.aspect_ratio),
  background = ($background),
  splats = {
    ($controls),
  },
}"
  | save splats.lua -f
}
